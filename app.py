#!/usr/bin/env python3
"""
Mental Rotation Task - Flask + Supabase + one-row-per-participant checkpoint storage

Storage model:
- Exactly ONE row per participant/session in public.sessions
- No per-trial database rows used by the app
- Checkpoints every 32 REAL trials append into responses_json in the same sessions row
- Resume from last saved checkpoint
- Completed participant_id is blocked from re-entering
- /data allows password-protected CSV download (one row per participant)

Environment variables required:
  SUPABASE_URL
  SUPABASE_SERVICE_ROLE_KEY

Optional environment variables:
  SECRET_KEY
  PORT
  DATA_EXPORT_PASSWORD   (defaults to HUT2026)
"""

from __future__ import annotations

import csv
import hmac
import io
import json
import os
import random
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from flask import (
    Flask,
    Response,
    jsonify,
    render_template_string,
    request,
    send_from_directory,
)

# ============================================================
# Config
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
IMGS_DIR = BASE_DIR / "imgs"
PRAC_DIR = BASE_DIR / "prac_imgs"

ATTN_N = 5
CHECKPOINT_SIZE = 32
SECRET_KEY = os.environ.get("SECRET_KEY", "replace-this-in-production")
PORT = int(os.environ.get("PORT", "5000"))
DATA_EXPORT_PASSWORD = os.environ.get("DATA_EXPORT_PASSWORD", "HUT2026")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()

if not SUPABASE_URL:
    raise RuntimeError("Missing SUPABASE_URL environment variable")
if not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("Missing SUPABASE_SERVICE_ROLE_KEY environment variable")

REST_BASE = f"{SUPABASE_URL}/rest/v1"

app = Flask(__name__)
app.config["SECRET_KEY"] = SECRET_KEY

# ============================================================
# Supabase helpers
# ============================================================

def supabase_headers(prefer: str | None = None) -> dict[str, str]:
    headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
    }
    if prefer:
        headers["Prefer"] = prefer
    return headers

def supabase_request(
    method: str,
    table_or_path: str,
    *,
    params: dict[str, Any] | None = None,
    json_data: Any = None,
    prefer: str | None = None,
    timeout: int = 30,
) -> requests.Response:
    url = table_or_path if table_or_path.startswith("http") else f"{REST_BASE}/{table_or_path.lstrip('/')}"
    resp = requests.request(
        method=method.upper(),
        url=url,
        headers=supabase_headers(prefer=prefer),
        params=params,
        json=json_data,
        timeout=timeout,
    )
    if not resp.ok:
        raise RuntimeError(f"Supabase error {resp.status_code}: {resp.text}")
    return resp

def supabase_insert(table: str, row: dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    resp = supabase_request("POST", table, json_data=row, prefer="return=representation")
    data = resp.json()
    return data if isinstance(data, list) else [data]

def supabase_select(
    table: str,
    *,
    select: str = "*",
    filters: dict[str, str] | None = None,
    order: str | None = None,
    limit: int | None = None,
    offset: int | None = None,
) -> list[dict[str, Any]]:
    params: dict[str, Any] = {"select": select}
    if filters:
        params.update(filters)
    if order:
        params["order"] = order
    if limit is not None:
        params["limit"] = str(limit)
    if offset is not None:
        params["offset"] = str(offset)
    return supabase_request("GET", table, params=params).json()

def supabase_update(
    table: str,
    *,
    filters: dict[str, str],
    patch: dict[str, Any],
) -> list[dict[str, Any]]:
    resp = supabase_request("PATCH", table, params=filters, json_data=patch, prefer="return=representation")
    data = resp.json()
    return data if isinstance(data, list) else [data]

def supabase_fetch_all(
    table: str,
    *,
    select: str = "*",
    filters: dict[str, str] | None = None,
    order: str | None = None,
    page_size: int = 1000,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    offset = 0
    while True:
        chunk = supabase_select(
            table,
            select=select,
            filters=filters,
            order=order,
            limit=page_size,
            offset=offset,
        )
        if not chunk:
            break
        out.extend(chunk)
        if len(chunk) < page_size:
            break
        offset += page_size
    return out

# ============================================================
# Stimulus helpers
# ============================================================

def list_pngs(folder: Path) -> list[str]:
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Missing folder: {folder.resolve()}")
    pngs = sorted([p.name for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".png"])
    if not pngs:
        raise FileNotFoundError(f"No .png files found in: {folder.resolve()}")
    return pngs

def list_stimuli() -> list[str]:
    return list_pngs(IMGS_DIR)

def list_practice() -> list[str]:
    return list_pngs(PRAC_DIR)

def is_reference(imgname: str) -> bool:
    return "reference" in imgname.lower()

def is_mirrored(imgname: str) -> bool:
    return "mirrored" in imgname.lower()

def correct_answer(imgname: str) -> str:
    if is_reference(imgname):
        return "mirrored" if is_mirrored(imgname) else "same"
    return "mirrored" if is_mirrored(imgname) else "normal"

_STIM_RE = re.compile(
    r"^(?P<letter>[A-Za-z]+)_(?P<kind>centered|reference)_(?P<state>normal|mirrored)_(?P<angle>\d{1,3})\.png$",
    re.IGNORECASE,
)

@dataclass(frozen=True)
class StimInfo:
    letter: str
    kind: str
    state: str
    angle: str

def parse_stim(img: str) -> StimInfo | None:
    m = _STIM_RE.match(img)
    if not m:
        return None
    return StimInfo(
        letter=m.group("letter").upper(),
        kind=m.group("kind").lower(),
        state=m.group("state").lower(),
        angle=f"{int(m.group('angle')):03d}",
    )

def build_real_trial_sequence(stimuli64: list[str]) -> list[dict[str, Any]]:
    seq: list[dict[str, Any]] = []
    order_index = 1
    for cycle in (1, 2, 3):
        perm = stimuli64[:]
        random.shuffle(perm)
        for img in perm:
            seq.append({
                "kind": "real",
                "dir": "imgs",
                "order_index": order_index,
                "cycle": cycle,
                "img": img,
            })
            order_index += 1
    return seq

def insert_attention_trials(real_trials: list[dict[str, Any]], stimuli64: list[str], n: int) -> list[dict[str, Any]]:
    if n <= 0:
        return real_trials[:]
    flow = real_trials[:]
    positions = sorted(random.sample(range(len(flow) + 1), k=n), reverse=True)
    for k, pos in enumerate(positions, start=1):
        img = random.choice(stimuli64)
        flow.insert(pos, {
            "kind": "attn",
            "dir": "imgs",
            "img": img,
            "attn_id": k,
        })
    return flow

def pick_practice_trials(prac_pngs: list[str]) -> list[dict[str, Any]]:
    def pick(where_kind: str, where_state: str) -> str:
        candidates = [p for p in prac_pngs if where_kind in p.lower() and where_state in p.lower()]
        if not candidates:
            raise FileNotFoundError(
                f"Could not find practice image matching '{where_kind}_{where_state}' in {PRAC_DIR.resolve()}"
            )
        return random.choice(candidates)

    return [
        {"kind": "prac", "dir": "prac_imgs", "img": pick("centered", "normal")},
        {"kind": "prac", "dir": "prac_imgs", "img": pick("centered", "mirrored")},
        {"kind": "prac", "dir": "prac_imgs", "img": pick("reference", "normal")},
        {"kind": "prac", "dir": "prac_imgs", "img": pick("reference", "mirrored")},
    ]

# ============================================================
# Session helpers
# ============================================================

def json_load_maybe(s: str | None, default: Any) -> Any:
    if not s:
        return default
    try:
        return json.loads(s)
    except Exception:
        return default

def get_session_by_participant(participant_id: str) -> dict[str, Any] | None:
    rows = supabase_select(
        "sessions",
        filters={"participant_id": f"eq.{participant_id}"},
        limit=1,
    )
    return rows[0] if rows else None

def get_session_by_id(session_id: str) -> dict[str, Any]:
    rows = supabase_select(
        "sessions",
        filters={"session_id": f"eq.{session_id}"},
        limit=1,
    )
    if not rows:
        raise ValueError("Invalid session")
    return rows[0]

def create_new_session(participant_id: str, return_url: str | None) -> dict[str, Any]:
    stimuli64 = list_stimuli()
    practice = pick_practice_trials(list_practice())
    real_trials = build_real_trial_sequence(stimuli64)
    flow = insert_attention_trials(real_trials, stimuli64, ATTN_N)

    session_id = uuid.uuid4().hex
    started_at = time.time()

    supabase_insert("sessions", {
        "session_id": session_id,
        "participant_id": participant_id,
        "return_url": return_url,
        "started_at_unix": started_at,
        "finished_at_unix": None,
        "status": "in_progress",
        "attention_correct": 0,
        "completed_real_trials": 0,
        "last_saved_pos": 0,
        "practice_json": json.dumps(practice),
        "flow_json": json.dumps(flow),
        "responses_json": json.dumps([]),
    })

    return {
        "mode": "new",
        "session_id": session_id,
        "participant_id": participant_id,
        "return_url": return_url,
        "practice": practice,
        "flow": flow,
        "resume_index": 0,
        "completed_real_trials": 0,
    }

def load_or_create_session(participant_id: str, return_url: str | None) -> dict[str, Any]:
    existing = get_session_by_participant(participant_id)

    if existing:
        if existing.get("status") == "finished":
            return {
                "mode": "completed",
                "message": "Your ID Has Already Cleared the Trials",
            }

        practice = json_load_maybe(existing.get("practice_json"), [])
        flow = json_load_maybe(existing.get("flow_json"), [])
        completed_real_trials = int(existing.get("completed_real_trials") or 0)
        resume_index = int(existing.get("last_saved_pos") or 0)

        if return_url and existing.get("return_url") != return_url:
            supabase_update(
                "sessions",
                filters={"session_id": f"eq.{existing['session_id']}"},
                patch={"return_url": return_url},
            )
            existing["return_url"] = return_url

        return {
            "mode": "resume",
            "session_id": existing["session_id"],
            "participant_id": participant_id,
            "return_url": existing.get("return_url"),
            "practice": practice,
            "flow": flow,
            "resume_index": resume_index,
            "completed_real_trials": completed_real_trials,
        }

    return create_new_session(participant_id, return_url)

def insert_checkpoint(session_id: str, trials: list[dict[str, Any]], current_pos: int) -> dict[str, Any]:
    sess = get_session_by_id(session_id)

    if sess.get("status") == "finished":
        raise ValueError("Session already finished")

    responses = json_load_maybe(sess.get("responses_json"), [])
    completed_real_trials = int(sess.get("completed_real_trials") or 0)
    attention_correct = int(sess.get("attention_correct") or 0)

    real_trials = [t for t in trials if str(t.get("kind")).lower() == "real"]
    unsaved_real_count = len(real_trials)

    if unsaved_real_count == 0:
        raise ValueError("Checkpoint contained no real trials")

    if unsaved_real_count != CHECKPOINT_SIZE and (completed_real_trials + unsaved_real_count) != 192:
        raise ValueError("Checkpoint must contain 32 real trials unless it is the final checkpoint.")

    expected_next = completed_real_trials + 1
    real_order_indices = sorted(int(t["order_index"]) for t in real_trials)

    if real_order_indices[0] != expected_next:
        raise ValueError(f"Expected next real order index {expected_next}, got {real_order_indices[0]}")

    if real_order_indices != list(range(real_order_indices[0], real_order_indices[0] + len(real_order_indices))):
        raise ValueError("Real trial order indices in checkpoint are not contiguous.")

    existing_real_order_indices = {
        int(t["order_index"])
        for t in responses
        if str(t.get("kind")).lower() == "real" and t.get("order_index") is not None
    }
    if any(oi in existing_real_order_indices for oi in real_order_indices):
        raise ValueError("Checkpoint includes already-saved real trials.")

    stamped_trials: list[dict[str, Any]] = []
    added_attention_correct = 0

    for tr in trials:
        kind = str(tr["kind"]).lower()
        img = str(tr["img"])
        answer = str(tr["answer"]).lower()
        time_ms = int(tr["time_ms"])

        saved = {
            "kind": kind,
            "order_index": int(tr["order_index"]) if tr.get("order_index") is not None else None,
            "cycle": int(tr["cycle"]) if tr.get("cycle") is not None else None,
            "img": img,
            "answer": answer,
            "time_ms": time_ms,
            "recorded_at_unix": time.time(),
        }

        if kind == "attn":
            saved["correct"] = 1 if answer == "same" else 0
            if saved["correct"] == 1:
                added_attention_correct += 1
        elif kind == "real":
            saved["correct"] = 1 if answer == correct_answer(img) else 0
        else:
            raise ValueError("Bad trial kind in checkpoint")

        stamped_trials.append(saved)

    responses.extend(stamped_trials)

    new_completed_real_trials = completed_real_trials + unsaved_real_count
    new_status = "finished" if new_completed_real_trials == 192 else "in_progress"
    finished_at = time.time() if new_status == "finished" else None

    patch = {
        "responses_json": json.dumps(responses),
        "completed_real_trials": new_completed_real_trials,
        "attention_correct": attention_correct + added_attention_correct,
        "last_saved_pos": int(current_pos),
        "status": new_status,
        "finished_at_unix": finished_at,
    }

    supabase_update(
        "sessions",
        filters={"session_id": f"eq.{session_id}"},
        patch=patch,
    )

    return {
        "ok": True,
        "completed_real_trials": new_completed_real_trials,
        "finished": new_status == "finished",
        "return_url": sess.get("return_url"),
    }

# ============================================================
# CSV export (ONE ROW PER PARTICIPANT)
# ============================================================

def build_export_csv() -> str:
    sessions = supabase_fetch_all("sessions", order="started_at_unix.asc")

    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "participant_id",
        "session_id",
        "started_at_unix",
        "finished_at_unix",
        "status",
        "attention_correct",
        "completed_real_trials",
        "last_saved_pos",
        "return_url",
        "practice_json",
        "flow_json",
        "responses_json",
    ])

    for sess in sessions:
        writer.writerow([
            sess.get("participant_id", ""),
            sess.get("session_id", ""),
            sess.get("started_at_unix", ""),
            sess.get("finished_at_unix", ""),
            sess.get("status", ""),
            sess.get("attention_correct", ""),
            sess.get("completed_real_trials", ""),
            sess.get("last_saved_pos", ""),
            sess.get("return_url", ""),
            sess.get("practice_json", ""),
            sess.get("flow_json", ""),
            sess.get("responses_json", ""),
        ])

    return output.getvalue()

# ============================================================
# Front-end
# ============================================================

INDEX_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Mental Rotation Task</title>
  <style>
    :root { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }
    body { margin: 0; background: #f6f6f7; color: #111; }
    .wrap { max-width: 900px; margin: 0 auto; padding: 28px 18px; }
    .card {
      background: #fff; border: 1px solid #e6e6ea; border-radius: 14px;
      padding: 22px; box-shadow: 0 4px 18px rgba(0,0,0,0.05);
    }
    h1,h2,h3 { margin: 0 0 12px; }
    p { margin: 10px 0; line-height: 1.35; color: #333; }
    ul { margin: 8px 0 14px 22px; color: #333; }
    li { margin: 6px 0; line-height: 1.35; }
    .center { text-align: center; }
    .btnrow { display: flex; gap: 12px; justify-content: center; flex-wrap: wrap; margin-top: 14px; }
    button {
      padding: 12px 16px; border-radius: 12px; border: 1px solid #d7d7de;
      background: #111; color: #fff; font-weight: 650; font-size: 16px; cursor: pointer;
      min-width: 160px;
    }
    button:disabled { opacity: 0.55; cursor: not-allowed; }
    .imgbox { display: flex; justify-content: center; margin: 12px 0 6px; }
    img.stim { max-width: 600px; width: 100%; height: auto; border-radius: 12px; border: 1px solid #e6e6ea; }
    .meta { display: flex; justify-content: space-between; gap: 10px; margin-top: 10px; color: #444; font-size: 14px; }
    .kbd { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; background: #f1f1f3; padding: 2px 6px; border-radius: 6px; border: 1px solid #e3e3e8;}
    .hidden { display: none; }
    .error { color: #b00020; }
    .instructions { max-width: 760px; margin: 0 auto; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card" id="screen-loading">
      <h1 class="center">Mental Rotation Task</h1>
      <p class="center">Loading…</p>
    </div>

    <div class="card hidden" id="screen-error">
      <h2 class="center">Unable to start task</h2>
      <p class="center error" id="error-message"></p>
    </div>

    <div class="card hidden" id="screen-locked">
      <h2 class="center">Task Complete</h2>
      <p class="center" id="locked-message">Your ID Has Already Cleared the Trials</p>
    </div>

    <div class="card hidden" id="screen-title">
      <h1 class="center">Mental Rotation Task</h1>
      <p class="center" id="title-message">Click start to begin.</p>
      <div class="btnrow">
        <button id="btn-start">Start</button>
      </div>
    </div>

    <div class="card hidden" id="screen-instructions">
      <div class="instructions">
        <h2 class="center">Instructions</h2>

        <p>In this task, you will see letters presented either:</p>
        <ul>
          <li>in pairs, or</li>
          <li>one at a time</li>
        </ul>

        <p>The letters may be rotated at different angles.</p>

        <h3>1. When TWO letters are shown:</h3>
        <p>Your task is to decide whether the letters are:</p>
        <ul>
          <li><strong>Same</strong> → the letters are identical, just rotated</li>
          <li><strong>Mirror</strong> → one letter is a flipped (mirror image) version of the other</li>
        </ul>
        <p>Rotation does not change whether letters are the same—only flipping does.</p>

        <h3>2. When ONE letter is shown:</h3>
        <p>Your task is to decide whether the letter is:</p>
        <ul>
          <li><strong>Normal</strong> → the letter is in its standard (non-mirrored) form</li>
          <li><strong>Mirrored</strong> → the letter is flipped</li>
        </ul>

        <h3>How to Respond:</h3>
        <ul>
          <li>Use your mouse to click the correct button on the screen</li>
          <li>Each trial will display the appropriate response options (e.g., Same / Mirror or Normal / Mirrored)</li>
          <li>Click the option that best matches your answer</li>
        </ul>

        <h3>General Guidelines:</h3>
        <ul>
          <li>Letters may appear at different rotations</li>
          <li>You may need to mentally rotate the letters to decide</li>
          <li>Respond as quickly and accurately as possible</li>
          <li>Try to keep your cursor near the response buttons to respond efficiently</li>
          <li>Stay focused and avoid distractions</li>
        </ul>

        <div class="btnrow">
          <button id="btn-begin-practice">Begin Practice Trials</button>
        </div>
      </div>
    </div>

    <div class="card hidden" id="screen-ready">
      <h2 class="center">Practice complete</h2>
      <p class="center">When you're ready, start the real task.</p>
      <div class="btnrow">
        <button id="btn-begin-test">Start Task</button>
      </div>
    </div>

    <div class="card hidden" id="screen-test">
      <h2 class="center" id="prompt">Prompt</h2>
      <div class="imgbox">
        <img class="stim" id="stim-img" alt="stimulus" />
      </div>
      <div class="btnrow" id="choices"></div>
      <div class="meta" id="meta-row">
        <div>Real trial: <span class="kbd" id="trial-idx">1</span> / <span class="kbd">192</span></div>
        <div>Cycle: <span class="kbd" id="cycle-idx">1</span> / <span class="kbd">3</span></div>
      </div>
    </div>

    <div class="card hidden" id="screen-done">
      <h2 class="center">Thank you!</h2>
      <p class="center" id="done-message">You may now close this window.</p>
      <div class="btnrow hidden" id="done-return-wrap">
        <button id="btn-return">Return to Survey</button>
      </div>
    </div>
  </div>

<script>
  const $ = (id) => document.getElementById(id);

  const screens = {
    loading: $("screen-loading"),
    error: $("screen-error"),
    locked: $("screen-locked"),
    title: $("screen-title"),
    instructions: $("screen-instructions"),
    ready: $("screen-ready"),
    test: $("screen-test"),
    done: $("screen-done")
  };

  function show(which) {
    for (const k of Object.keys(screens)) screens[k].classList.add("hidden");
    screens[which].classList.remove("hidden");
  }

  function setError(msg) {
    $("error-message").textContent = msg;
    show("error");
  }

  function api(path, method="GET", body=null) {
    const opts = { method, headers: {} };
    if (body !== null) {
      opts.headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(body);
    }
    return fetch(path, opts).then(async (res) => {
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(`HTTP ${res.status}: ${txt}`);
      }
      return res.json();
    });
  }

  function imgSrc(item) {
    const dir = item.dir || "imgs";
    if (dir === "prac_imgs") return `/prac_imgs/${encodeURIComponent(item.img)}`;
    return `/imgs/${encodeURIComponent(item.img)}`;
  }

  function clearChoices() {
    $("choices").innerHTML = "";
  }

  function addChoiceButton(label, value, onClick) {
    const b = document.createElement("button");
    b.textContent = label;
    b.addEventListener("click", onClick);
    $("choices").appendChild(b);
  }

  function setButtonsEnabled(enabled) {
    $("choices").querySelectorAll("button").forEach(b => { b.disabled = !enabled; });
  }

  function practiceTitleText(item) {
    const name = (item.img || "").toLowerCase();
    if (name.includes("mirrored")) return "This text is Mirrored";
    if (name.includes("reference")) return "This text is Same";
    return "This text is Normal";
  }

  let sessionId = null;
  let participantId = null;
  let returnUrl = null;
  let practice = [];
  let practicePos = 0;
  let flow = [];
  let pos = 0;
  let t0 = 0;
  let completedRealTrials = 0;
  let unsavedChunk = [];
  let currentChunkStartPos = 0;

  function renderPractice(item) {
    $("meta-row").style.display = "none";
    $("stim-img").src = imgSrc(item);
    $("prompt").textContent = practiceTitleText(item);
    clearChoices();

    const isReference = (item.img || "").toLowerCase().includes("reference");
    if (isReference) {
      addChoiceButton("Same", "same", () => advancePractice());
      addChoiceButton("Mirrored", "mirrored", () => advancePractice());
    } else {
      addChoiceButton("Normal", "normal", () => advancePractice());
      addChoiceButton("Mirrored", "mirrored", () => advancePractice());
    }
  }

  function advancePractice() {
    practicePos += 1;
    if (practicePos >= practice.length) {
      show("ready");
      return;
    }
    renderPractice(practice[practicePos]);
  }

  function renderTrial(item) {
    $("meta-row").style.display = "";
    const currentReal = item.kind === "real"
      ? item.order_index
      : Math.min(192, completedRealTrials + unsavedChunk.filter(x => x.kind === "real").length + 1);

    $("trial-idx").textContent = String(currentReal);
    $("cycle-idx").textContent = item.kind === "real" ? String(item.cycle) : "-";
    $("stim-img").src = imgSrc(item);
    clearChoices();

    if (item.kind === "attn") {
      $("prompt").textContent = "Click Option Same";
      addChoiceButton("Same", "same", () => submitAnswer("same"));
      addChoiceButton("Mirrored", "mirrored", () => submitAnswer("mirrored"));
    } else {
      const isReference = (item.img || "").toLowerCase().includes("reference");
      if (isReference) {
        $("prompt").textContent = "Are these the same letter?";
        addChoiceButton("Same", "same", () => submitAnswer("same"));
        addChoiceButton("Mirrored", "mirrored", () => submitAnswer("mirrored"));
      } else {
        $("prompt").textContent = "Is this letter normal or mirrored?";
        addChoiceButton("Normal", "normal", () => submitAnswer("normal"));
        addChoiceButton("Mirrored", "mirrored", () => submitAnswer("mirrored"));
      }
    }

    t0 = performance.now();
    setButtonsEnabled(true);
  }

  async function saveChunk() {
    if (!unsavedChunk.length) {
      return { ok: true, completed_real_trials: completedRealTrials, finished: false };
    }

    const payload = {
      session_id: sessionId,
      trials: unsavedChunk,
      current_pos: pos
    };

    const resp = await api("/api/checkpoint", "POST", payload);
    completedRealTrials = resp.completed_real_trials;
    unsavedChunk = [];
    currentChunkStartPos = pos;
    return resp;
  }

  async function submitAnswer(value) {
    if (!sessionId) return;
    setButtonsEnabled(false);

    const item = flow[pos];
    const dt = Math.max(0, Math.round(performance.now() - t0));

    unsavedChunk.push({
      kind: item.kind,
      order_index: item.kind === "real" ? item.order_index : null,
      cycle: item.kind === "real" ? item.cycle : null,
      img: item.img,
      answer: value,
      time_ms: dt
    });

    pos += 1;

    const unsavedReal = unsavedChunk.filter(x => x.kind === "real").length;
    const totalRealAfterThis = completedRealTrials + unsavedReal;
    const needsCheckpoint = (unsavedReal === 32) || (totalRealAfterThis === 192);

    if (needsCheckpoint) {
      try {
        const resp = await saveChunk();

        if (resp.finished) {
          $("done-return-wrap").classList.add("hidden");
          $("done-message").textContent = "You may now close this window.";
          if (resp.return_url) {
            returnUrl = resp.return_url;
            $("done-message").textContent = "Your responses were saved. Click below to return to the survey.";
            $("done-return-wrap").classList.remove("hidden");
          }
          show("done");
          return;
        }
      } catch (e) {
        console.error(e);
        pos = currentChunkStartPos;
        unsavedChunk = [];
        setError("There was a problem saving at the checkpoint. You have been returned to the last saved checkpoint.");
        return;
      }
    }

    if (pos >= flow.length) {
      show("done");
      return;
    }

    renderTrial(flow[pos]);
  }

  $("btn-start").addEventListener("click", async () => {
    try {
      const resp = await api("/api/start", "POST", {
        participant_id: participantId,
        return_url: returnUrl
      });

      if (resp.mode === "completed") {
        $("locked-message").textContent = resp.message || "Your ID Has Already Cleared the Trials";
        show("locked");
        return;
      }

      sessionId = resp.session_id;
      practice = resp.practice || [];
      flow = resp.flow || [];
      completedRealTrials = resp.completed_real_trials || 0;
      pos = resp.resume_index || 0;
      currentChunkStartPos = pos;
      practicePos = 0;
      unsavedChunk = [];

      if (resp.mode === "resume") {
        $("title-message").textContent = `Resuming from checkpoint after real trial ${completedRealTrials}. Click start to continue.`;
      } else {
        $("title-message").textContent = "Click start to begin.";
      }

      if (resp.mode === "resume") {
        show("ready");
      } else {
        show("instructions");
      }
    } catch (e) {
      console.error(e);
      setError("Could not start the task.");
    }
  });

  $("btn-begin-practice").addEventListener("click", () => {
    if (!practice.length) {
      show("ready");
      return;
    }
    show("test");
    renderPractice(practice[practicePos]);
  });

  $("btn-begin-test").addEventListener("click", () => {
    if (!flow.length) {
      setError("No trials were loaded.");
      return;
    }
    if (pos >= flow.length) {
      show("done");
      return;
    }
    show("test");
    renderTrial(flow[pos]);
  });

  $("btn-return").addEventListener("click", () => {
    if (returnUrl) window.location.href = returnUrl;
  });

  function initFromURL() {
    const params = new URLSearchParams(window.location.search);
    participantId = (params.get("participant_id") || "").trim();
    returnUrl = (params.get("return_url") || "").trim() || null;

    if (!participantId) {
      setError("Missing participant_id. This task must be launched from Qualtrics.");
      return;
    }

    show("title");
  }

  initFromURL();
</script>
</body>
</html>
"""

DATA_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Download Data</title>
  <style>
    :root { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }
    body { margin: 0; background: #f6f6f7; color: #111; }
    .wrap { max-width: 480px; margin: 0 auto; padding: 40px 18px; }
    .card {
      background: #fff; border: 1px solid #e6e6ea; border-radius: 14px;
      padding: 24px; box-shadow: 0 4px 18px rgba(0,0,0,0.05);
    }
    input[type=password] {
      width: 100%; padding: 12px; border-radius: 10px; border: 1px solid #d7d7de;
      margin-top: 10px; box-sizing: border-box; font-size: 16px;
    }
    button {
      margin-top: 14px; padding: 12px 16px; border-radius: 12px; border: 1px solid #d7d7de;
      background: #111; color: #fff; font-weight: 650; font-size: 16px; cursor: pointer;
      width: 100%;
    }
    .error { color: #b00020; margin-top: 10px; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h2>Download Data</h2>
      <p>Enter the password to download all saved data as CSV.</p>
      <form method="post">
        <input type="password" name="password" autocomplete="current-password" required />
        <button type="submit">Download CSV</button>
      </form>
      {% if error %}
        <div class="error">{{ error }}</div>
      {% endif %}
    </div>
  </div>
</body>
</html>
"""

# ============================================================
# Routes
# ============================================================

@app.route("/")
def index():
    return render_template_string(INDEX_HTML)

@app.route("/imgs/<path:filename>")
def serve_imgs(filename: str):
    return send_from_directory(IMGS_DIR, filename)

@app.route("/prac_imgs/<path:filename>")
def serve_prac_imgs(filename: str):
    return send_from_directory(PRAC_DIR, filename)

@app.route("/health")
def health():
    return jsonify({"ok": True})

@app.route("/data", methods=["GET", "POST"])
def data_download():
    if request.method == "GET":
        return render_template_string(DATA_HTML, error=None)

    submitted = str(request.form.get("password", ""))
    if not hmac.compare_digest(submitted, DATA_EXPORT_PASSWORD):
        return render_template_string(DATA_HTML, error="Incorrect password.")

    csv_text = build_export_csv()
    ts = time.strftime("%Y%m%d-%H%M%S")
    filename = f"mental_rotation_data_{ts}.csv"

    return Response(
        csv_text,
        mimetype="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )

@app.route("/api/start", methods=["POST"])
def api_start():
    data = request.get_json(silent=True) or {}
    participant_id = str(data.get("participant_id", "")).strip()
    return_url = data.get("return_url", None)
    if return_url is not None:
        return_url = str(return_url).strip() or None

    if not participant_id:
        return "Missing participant_id", 400

    try:
        session_data = load_or_create_session(participant_id, return_url)
    except Exception as e:
        return f"Failed to start session: {e}", 500

    return jsonify(session_data)

@app.route("/api/checkpoint", methods=["POST"])
def api_checkpoint():
    data = request.get_json(silent=True) or {}
    session_id = str(data.get("session_id", "")).strip()
    trials = data.get("trials", [])
    current_pos = data.get("current_pos", None)

    if not session_id:
        return "Missing session_id", 400
    if not isinstance(trials, list) or not trials:
        return "Missing trials", 400
    if current_pos is None:
        return "Missing current_pos", 400

    try:
        current_pos = int(current_pos)
        result = insert_checkpoint(session_id, trials, current_pos)
    except ValueError as e:
        return str(e), 400
    except Exception as e:
        return f"Failed to save checkpoint: {e}", 500

    return jsonify(result)

# ============================================================
# Startup validation
# ============================================================

def validate_assets() -> None:
    stimuli = list_stimuli()
    if len(stimuli) != 64:
        print(f"[warn] Found {len(stimuli)} .png files in imgs/ (expected 64). The app will still run.")
    _ = list_practice()

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    validate_assets()
    print(f"Serving on http://127.0.0.1:{PORT}")
    print(f"Images directory:   {IMGS_DIR}")
    print(f"Practice directory: {PRAC_DIR}")
    print(f"Supabase URL:       {SUPABASE_URL}")
    app.run(host="0.0.0.0", port=PORT, debug=True)
