#!/usr/bin/env python3
"""
Mental Rotation Task - Flask + SQLite + CSV export
Designed for Qualtrics -> hosted web app -> save data -> optional redirect back.

Folder layout:
  app.py
  imgs/         (contains 64 real stimuli .png files)
  prac_imgs/    (contains practice .png files)

Outputs:
  data.db       (SQLite database; server-side storage)
  data.csv      (one summary row per completed participant, similar to prior workflow)

Qualtrics launch example:
  https://your-app.onrender.com/?participant_id=${e://Field/participant_id}

Optional return URL:
  https://your-app.onrender.com/?participant_id=${e://Field/participant_id}&return_url=https%3A%2F%2Fyour-qualtrics-return-link

Notes:
- participant_id is required from the URL.
- return_url is optional.
- practice is not timed/scored/recorded.
- real task preserves 3 cycles of the 64 stimuli = 192 real trials.
- 5 attention checks are extra trials.
- detailed data is saved in SQLite.
- on finish, a summary row is appended to CSV as well.
"""

from __future__ import annotations

import csv
import os
import random
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flask import (
    Flask,
    jsonify,
    redirect,
    render_template_string,
    request,
    send_from_directory,
    url_for,
)

# ============================================================
# Config
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
IMGS_DIR = BASE_DIR / "imgs"
PRAC_DIR = BASE_DIR / "prac_imgs"

DATA_DIR = Path(os.environ.get("DATA_DIR", BASE_DIR))
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "data.db"
CSV_PATH = DATA_DIR / "data.csv"

ATTN_N = 5
SECRET_KEY = os.environ.get("SECRET_KEY", "replace-this-in-production")
PORT = int(os.environ.get("PORT", "5000"))

app = Flask(__name__)
app.config["SECRET_KEY"] = SECRET_KEY

# ============================================================
# Helpers
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

def is_centered(imgname: str) -> bool:
    return "centered" in imgname.lower()

def is_mirrored(imgname: str) -> bool:
    return "mirrored" in imgname.lower()

def is_normal(imgname: str) -> bool:
    return "normal" in imgname.lower()

def correct_answer(imgname: str) -> str:
    """
    Answer vocabulary:
      - reference images: "same" or "mirrored"
      - centered images:  "normal" or "mirrored"
    """
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
    letter = m.group("letter").upper()
    kind = m.group("kind").lower()
    state = m.group("state").lower()
    angle = m.group("angle")
    try:
        angle_i = int(angle)
        angle = f"{angle_i:03d}"
    except Exception:
        pass
    return StimInfo(letter=letter, kind=kind, state=state, angle=angle)

def build_real_trial_sequence(stimuli64: list[str]) -> list[dict[str, Any]]:
    """
    3 cycles. Each cycle is a random permutation of the 64 stimuli.
    Returns 192 REAL trial dicts.
    """
    seq: list[dict[str, Any]] = []
    order = 1
    for cycle in (1, 2, 3):
        perm = stimuli64[:]
        random.shuffle(perm)
        for img in perm:
            seq.append({
                "kind": "real",
                "dir": "imgs",
                "order_index": order,
                "cycle": cycle,
                "img": img,
            })
            order += 1
    return seq

def insert_attention_trials(real_trials: list[dict[str, Any]], stimuli64: list[str], n: int) -> list[dict[str, Any]]:
    """
    Insert n attention checks randomly throughout the flow.
    """
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
    """
    Exactly 4 practice images in this order:
      1) centered_normal
      2) centered_mirrored
      3) reference_normal
      4) reference_mirrored
    Angle can be random.
    """
    def pick(where_kind: str, where_state: str) -> str:
        candidates = []
        for p in prac_pngs:
            low = p.lower()
            if where_kind in low and where_state in low:
                candidates.append(p)
        if not candidates:
            raise FileNotFoundError(
                f"Could not find practice image matching '{where_kind}_{where_state}' in {PRAC_DIR.resolve()}"
            )
        return random.choice(candidates)

    img1 = pick("centered", "normal")
    img2 = pick("centered", "mirrored")
    img3 = pick("reference", "normal")
    img4 = pick("reference", "mirrored")

    return [
        {"kind": "prac", "dir": "prac_imgs", "img": img1},
        {"kind": "prac", "dir": "prac_imgs", "img": img2},
        {"kind": "prac", "dir": "prac_imgs", "img": img3},
        {"kind": "prac", "dir": "prac_imgs", "img": img4},
    ]

# ============================================================
# Block A aggregation
# ============================================================

def build_canonical_map(stimuli_sorted: list[str]) -> dict[tuple[str, str, str], str]:
    """
    Map (letter, angle, kind) -> canonical filename.
    Canonical preference: normal > anything else.
    """
    bucket: dict[tuple[str, str, str], list[str]] = {}
    for img in stimuli_sorted:
        info = parse_stim(img)
        if not info:
            continue
        if info.letter not in ("R", "G"):
            continue
        if info.kind not in ("centered", "reference"):
            continue
        key = (info.letter, info.angle, info.kind)
        bucket.setdefault(key, []).append(img)

    canon: dict[tuple[str, str, str], str] = {}
    for key, imgs in bucket.items():
        normals = [x for x in imgs if is_normal(x)]
        chosen = sorted(normals)[0] if normals else sorted(imgs)[0]
        canon[key] = chosen
    return canon

def angles_for_letter(canon: dict[tuple[str, str, str], str], letter: str) -> list[str]:
    return sorted({angle for (L, angle, kind) in canon.keys() if L == letter and kind in ("centered", "reference")})

def build_block_a_headers(stimuli_sorted: list[str]) -> list[str]:
    canon = build_canonical_map(stimuli_sorted)
    cols: list[str] = []

    for letter in ("R", "G"):
        for angle in angles_for_letter(canon, letter):
            base = f"{letter}_{angle}"
            cols.append(f"{base}_mean_time")
            cols.append(f"{base}_mean_time_scored")
            cols.append(f"{base}_score")

            cols.append(f"reference_{base}_mean_time")
            cols.append(f"reference_{base}_mean_time_scored")
            cols.append(f"reference_{base}_score")

            cols.append(f"centered_{base}_mean_time")
            cols.append(f"centered_{base}_mean_time_scored")
            cols.append(f"centered_{base}_score")

    for letter in ("R", "G"):
        cols.append(f"{letter}_mean_time")
        cols.append(f"{letter}_mean_time_scored")
        cols.append(f"{letter}_score")

        cols.append(f"reference_{letter}_mean_time")
        cols.append(f"reference_{letter}_mean_time_scored")
        cols.append(f"reference_{letter}_score")

        cols.append(f"centered_{letter}_mean_time")
        cols.append(f"centered_{letter}_mean_time_scored")
        cols.append(f"centered_{letter}_score")

    return cols

def mean_or_blank(values: list[int]) -> str:
    if not values:
        return ""
    return f"{(sum(values) / len(values)):.6g}"

def compute_block_a_from_trials(stimuli_sorted: list[str], ident_times: dict[tuple[int, str], int], ident_corr: dict[tuple[int, str], int]) -> dict[str, str | int]:
    canon = build_canonical_map(stimuli_sorted)
    out: dict[str, str | int] = {}

    def collect(imgs: list[str]) -> tuple[list[int], list[int]]:
        times: list[int] = []
        corrs: list[int] = []
        for cycle in (1, 2, 3):
            for img in imgs:
                t = ident_times.get((cycle, img))
                c = ident_corr.get((cycle, img))
                if t is None or c is None:
                    continue
                times.append(int(t))
                corrs.append(int(c))
        return times, corrs

    def write_mean_score(prefix: str, times: list[int], corrs: list[int]) -> None:
        out[f"{prefix}_mean_time"] = mean_or_blank(times)
        scored_times = [t for (t, c) in zip(times, corrs) if c == 1]
        out[f"{prefix}_mean_time_scored"] = mean_or_blank(scored_times)
        out[f"{prefix}_score"] = int(sum(corrs)) if corrs else 0

    for letter in ("R", "G"):
        for angle in angles_for_letter(canon, letter):
            ref_img = canon.get((letter, angle, "reference"))
            cen_img = canon.get((letter, angle, "centered"))

            ref_imgs = [ref_img] if ref_img else []
            cen_imgs = [cen_img] if cen_img else []
            both_imgs = ref_imgs + cen_imgs

            times, corrs = collect(both_imgs)
            write_mean_score(f"{letter}_{angle}", times, corrs)

            times, corrs = collect(ref_imgs)
            write_mean_score(f"reference_{letter}_{angle}", times, corrs)

            times, corrs = collect(cen_imgs)
            write_mean_score(f"centered_{letter}_{angle}", times, corrs)

    for letter in ("R", "G"):
        ref_imgs: list[str] = []
        cen_imgs: list[str] = []
        for angle in angles_for_letter(canon, letter):
            ri = canon.get((letter, angle, "reference"))
            ci = canon.get((letter, angle, "centered"))
            if ri:
                ref_imgs.append(ri)
            if ci:
                cen_imgs.append(ci)

        both_imgs = ref_imgs + cen_imgs

        times, corrs = collect(both_imgs)
        write_mean_score(letter, times, corrs)

        times, corrs = collect(ref_imgs)
        write_mean_score(f"reference_{letter}", times, corrs)

        times, corrs = collect(cen_imgs)
        write_mean_score(f"centered_{letter}", times, corrs)

    return out

# ============================================================
# CSV header / export
# ============================================================

def build_csv_header(stimuli64_sorted: list[str]) -> list[str]:
    header: list[str] = [
        "participant_id",
        "session_id",
        "started_at_unix",
        "finished_at_unix",
        "attention",
    ]

    for cycle in (1, 2, 3):
        for img in stimuli64_sorted:
            header.append(f"c{cycle}_{img}_time_ms")

    for cycle in (1, 2, 3):
        for img in stimuli64_sorted:
            header.append(f"c{cycle}_{img}_correct")

    for i in range(1, 193):
        header.append(f"order_{i}_time_ms")

    for i in range(1, 193):
        header.append(f"order_{i}_correct")

    header.extend(build_block_a_headers(stimuli64_sorted))
    return header

def ensure_csv_exists(header: list[str]) -> None:
    if CSV_PATH.exists():
        return
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)

# ============================================================
# SQLite
# ============================================================

def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    conn = get_conn()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                participant_id TEXT NOT NULL,
                return_url TEXT,
                started_at_unix REAL NOT NULL,
                finished_at_unix REAL,
                status TEXT NOT NULL,
                attention_correct INTEGER DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trials (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                participant_id TEXT NOT NULL,
                trial_kind TEXT NOT NULL,
                order_index INTEGER,
                cycle INTEGER,
                img TEXT NOT NULL,
                answer TEXT NOT NULL,
                correct INTEGER NOT NULL,
                time_ms INTEGER NOT NULL,
                recorded_at_unix REAL NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(session_id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS session_payloads (
                session_id TEXT PRIMARY KEY,
                stimuli_sorted_json TEXT NOT NULL,
                practice_json TEXT NOT NULL,
                flow_json TEXT NOT NULL,
                real_trials_json TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(session_id)
            )
        """)
        conn.commit()
    finally:
        conn.close()

# ============================================================
# Session data helpers
# ============================================================

def create_session(participant_id: str, return_url: str | None) -> dict[str, Any]:
    stimuli64 = list_stimuli()
    stimuli_sorted = sorted(stimuli64)
    prac_pngs = list_practice()
    practice = pick_practice_trials(prac_pngs)
    real_trials = build_real_trial_sequence(stimuli64)
    flow = insert_attention_trials(real_trials, stimuli64, ATTN_N)

    session_id = uuid.uuid4().hex
    started_at = time.time()

    import json

    conn = get_conn()
    try:
        conn.execute("""
            INSERT INTO sessions (session_id, participant_id, return_url, started_at_unix, status, attention_correct)
            VALUES (?, ?, ?, ?, 'in_progress', 0)
        """, (session_id, participant_id, return_url, started_at))

        conn.execute("""
            INSERT INTO session_payloads (session_id, stimuli_sorted_json, practice_json, flow_json, real_trials_json)
            VALUES (?, ?, ?, ?, ?)
        """, (
            session_id,
            json.dumps(stimuli_sorted),
            json.dumps(practice),
            json.dumps(flow),
            json.dumps(real_trials),
        ))
        conn.commit()
    finally:
        conn.close()

    return {
        "session_id": session_id,
        "participant_id": participant_id,
        "return_url": return_url,
        "stimuli_sorted": stimuli_sorted,
        "practice": practice,
        "flow": flow,
        "real_trials": real_trials,
    }

def load_session_payload(session_id: str) -> dict[str, Any]:
    import json

    conn = get_conn()
    try:
        sess = conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if not sess:
            raise ValueError("Session not found")

        payload = conn.execute(
            "SELECT * FROM session_payloads WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if not payload:
            raise ValueError("Session payload not found")

        return {
            "session_id": sess["session_id"],
            "participant_id": sess["participant_id"],
            "return_url": sess["return_url"],
            "started_at_unix": sess["started_at_unix"],
            "finished_at_unix": sess["finished_at_unix"],
            "status": sess["status"],
            "attention_correct": sess["attention_correct"],
            "stimuli_sorted": json.loads(payload["stimuli_sorted_json"]),
            "practice": json.loads(payload["practice_json"]),
            "flow": json.loads(payload["flow_json"]),
            "real_trials": json.loads(payload["real_trials_json"]),
        }
    finally:
        conn.close()

def record_trial(session_id: str, kind: str, order_index: int | None, cycle: int | None, img: str, answer: str, time_ms: int) -> None:
    conn = get_conn()
    try:
        sess = conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if not sess:
            raise ValueError("Invalid session")
        if sess["status"] == "finished":
            raise ValueError("Session already finished")

        participant_id = sess["participant_id"]

        if kind == "attn":
            correct = 1 if answer == "same" else 0
            conn.execute("""
                INSERT INTO trials (
                    session_id, participant_id, trial_kind, order_index, cycle, img,
                    answer, correct, time_ms, recorded_at_unix
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id, participant_id, kind, None, None, img,
                answer, correct, time_ms, time.time()
            ))
            if correct == 1:
                conn.execute("""
                    UPDATE sessions
                    SET attention_correct = attention_correct + 1
                    WHERE session_id = ?
                """, (session_id,))
            conn.commit()
            return

        if kind != "real":
            raise ValueError("Bad trial kind")

        if order_index is None or cycle is None:
            raise ValueError("Missing order_index/cycle")

        correct = 1 if answer == correct_answer(img) else 0

        already = conn.execute("""
            SELECT 1 FROM trials
            WHERE session_id = ? AND trial_kind = 'real' AND order_index = ?
        """, (session_id, order_index)).fetchone()

        if already:
            raise ValueError("This real trial was already recorded")

        conn.execute("""
            INSERT INTO trials (
                session_id, participant_id, trial_kind, order_index, cycle, img,
                answer, correct, time_ms, recorded_at_unix
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id, participant_id, kind, order_index, cycle, img,
            answer, correct, time_ms, time.time()
        ))
        conn.commit()
    finally:
        conn.close()

def finish_session(session_id: str) -> dict[str, Any]:
    payload = load_session_payload(session_id)
    stimuli_sorted = payload["stimuli_sorted"]

    conn = get_conn()
    try:
        sess = conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if not sess:
            raise ValueError("Invalid session")

        if sess["status"] == "finished":
            return {
                "ok": True,
                "already": True,
                "return_url": sess["return_url"],
            }

        real_trials = conn.execute("""
            SELECT * FROM trials
            WHERE session_id = ? AND trial_kind = 'real'
            ORDER BY order_index ASC
        """, (session_id,)).fetchall()

        if len(real_trials) != 192:
            raise ValueError(f"Not all 192 real trials were recorded. Found {len(real_trials)}.")

        attn_trials = conn.execute("""
            SELECT * FROM trials
            WHERE session_id = ? AND trial_kind = 'attn'
            ORDER BY id ASC
        """, (session_id,)).fetchall()

        ident_times: dict[tuple[int, str], int] = {}
        ident_corr: dict[tuple[int, str], int] = {}
        order_times: list[int | None] = [None] * 192
        order_corr: list[int | None] = [None] * 192

        for tr in real_trials:
            cyc = int(tr["cycle"])
            img = str(tr["img"])
            tms = int(tr["time_ms"])
            cor = int(tr["correct"])
            oi = int(tr["order_index"])

            ident_times[(cyc, img)] = tms
            ident_corr[(cyc, img)] = cor
            order_times[oi - 1] = tms
            order_corr[oi - 1] = cor

        if any(v is None for v in order_times) or any(v is None for v in order_corr):
            raise ValueError("Order logging incomplete.")

        header = build_csv_header(stimuli_sorted)
        ensure_csv_exists(header)

        row: dict[str, Any] = {}
        row["participant_id"] = sess["participant_id"]
        row["session_id"] = sess["session_id"]
        row["started_at_unix"] = sess["started_at_unix"]
        row["finished_at_unix"] = time.time()
        row["attention"] = sess["attention_correct"]

        for cycle in (1, 2, 3):
            for img in stimuli_sorted:
                row[f"c{cycle}_{img}_time_ms"] = ident_times.get((cycle, img), "")

        for cycle in (1, 2, 3):
            for img in stimuli_sorted:
                row[f"c{cycle}_{img}_correct"] = ident_corr.get((cycle, img), "")

        for i in range(1, 193):
            row[f"order_{i}_time_ms"] = order_times[i - 1]

        for i in range(1, 193):
            row[f"order_{i}_correct"] = order_corr[i - 1]

        block_a = compute_block_a_from_trials(stimuli_sorted, ident_times, ident_corr)
        row.update(block_a)

        with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([row.get(col, "") for col in header])

        conn.execute("""
            UPDATE sessions
            SET finished_at_unix = ?, status = 'finished'
            WHERE session_id = ?
        """, (row["finished_at_unix"], session_id))
        conn.commit()

        return {
            "ok": True,
            "already": False,
            "return_url": sess["return_url"],
        }
    finally:
        conn.close()

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
    h1,h2 { margin: 0 0 12px; }
    p { margin: 10px 0; line-height: 1.35; color: #333; }
    .center { text-align: center; }
    .btnrow { display: flex; gap: 12px; justify-content: center; flex-wrap: wrap; margin-top: 14px; }
    button {
      padding: 12px 16px; border-radius: 12px; border: 1px solid #d7d7de;
      background: #111; color: #fff; font-weight: 650; font-size: 16px; cursor: pointer;
      min-width: 160px;
    }
    button.secondary { background: #fff; color: #111; }
    button:disabled { opacity: 0.55; cursor: not-allowed; }
    .imgbox { display: flex; justify-content: center; margin: 12px 0 6px; }
    img.stim { max-width: 600px; width: 100%; height: auto; border-radius: 12px; border: 1px solid #e6e6ea; }
    .meta { display: flex; justify-content: space-between; gap: 10px; margin-top: 10px; color: #444; font-size: 14px; }
    .kbd { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; background: #f1f1f3; padding: 2px 6px; border-radius: 6px; border: 1px solid #e3e3e8;}
    .hidden { display: none; }
    .error { color: #b00020; }
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

    <div class="card hidden" id="screen-title">
      <h1 class="center">Mental Rotation Task</h1>
      <p class="center">Click start to begin.</p>
      <div class="btnrow">
        <button id="btn-start">Start</button>
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
        <div>Trial: <span class="kbd" id="trial-idx">1</span> / <span class="kbd">192</span></div>
        <div>Cycle: <span class="kbd" id="cycle-idx">1</span> / <span class="kbd">3</span></div>
      </div>
    </div>

    <div class="card hidden" id="screen-done">
      <h2 class="center">Thank you for participating in this research endeavor!</h2>
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
    title: $("screen-title"),
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
    $("choices").querySelectorAll("button").forEach(b => {
      b.disabled = !enabled;
    });
  }

  function practiceTitleText(item) {
    const name = (item.img || "").toLowerCase();
    const mirrored = name.includes("mirrored");
    const reference = name.includes("reference");
    if (mirrored) return "This text is Mirrored";
    if (reference) return "This text is Same";
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

    if (item.kind === "real") {
      $("trial-idx").textContent = String(item.order_index);
      $("cycle-idx").textContent = String(item.cycle);
    }

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

  async function submitAnswer(value) {
    if (!sessionId) return;
    setButtonsEnabled(false);

    const item = flow[pos];
    const dt = Math.max(0, Math.round(performance.now() - t0));

    try {
      await api("/api/record", "POST", {
        session_id: sessionId,
        kind: item.kind,
        order_index: item.kind === "real" ? item.order_index : null,
        cycle: item.kind === "real" ? item.cycle : null,
        img: item.img,
        answer: value,
        time_ms: dt
      });
    } catch (e) {
      console.error(e);
      setError("There was a problem saving your response. Please do not refresh. Contact the researcher.");
      return;
    }

    pos += 1;
    if (pos >= flow.length) {
      try {
        const resp = await api("/api/finish", "POST", { session_id: sessionId });

        $("done-return-wrap").classList.add("hidden");
        $("done-message").textContent = "You may now close this window.";

        if (resp.return_url) {
          returnUrl = resp.return_url;
          $("done-message").textContent = "Your responses were saved. Click below to return to the survey.";
          $("done-return-wrap").classList.remove("hidden");
        }
      } catch (e) {
        console.error(e);
        $("done-message").textContent = "The task ended, but there was a problem finalizing your session. Please contact the researcher.";
      }

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

      sessionId = resp.session_id;
      practice = resp.practice || [];
      flow = resp.flow || [];
      practicePos = 0;
      pos = 0;

      if (!practice.length) {
        show("ready");
      } else {
        show("test");
        renderPractice(practice[practicePos]);
      }
    } catch (e) {
      console.error(e);
      setError("Could not start the task.");
    }
  });

  $("btn-begin-test").addEventListener("click", () => {
    if (!flow.length) {
      setError("No trials were loaded.");
      return;
    }
    show("test");
    renderTrial(flow[pos]);
  });

  $("btn-return").addEventListener("click", () => {
    if (returnUrl) {
      window.location.href = returnUrl;
    }
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

@app.route("/api/start", methods=["POST"])
def api_start():
    data = request.get_json(silent=True) or {}
    participant_id = str(data.get("participant_id", "")).strip()
    return_url = data.get("return_url", None)
    if return_url is not None:
        return_url = str(return_url).strip() or None

    if not participant_id:
        return "Missing participant_id", 400

    session_data = create_session(participant_id, return_url)
    return jsonify({
        "session_id": session_data["session_id"],
        "practice": session_data["practice"],
        "flow": session_data["flow"],
    })

@app.route("/api/record", methods=["POST"])
def api_record():
    data = request.get_json(silent=True) or {}

    session_id = str(data.get("session_id", "")).strip()
    kind = str(data.get("kind", "")).strip().lower()
    img = str(data.get("img", "")).strip()
    answer = str(data.get("answer", "")).strip().lower()

    time_ms_raw = data.get("time_ms", None)
    order_index = data.get("order_index", None)
    cycle = data.get("cycle", None)

    if not session_id or not img or time_ms_raw is None:
        return "Bad record payload", 400

    try:
        time_ms = int(time_ms_raw)
    except Exception:
        return "time_ms must be int", 400

    if time_ms < 0:
        return "time_ms must be nonnegative", 400

    if kind not in ("real", "attn"):
        return "Bad kind", 400

    try:
        if order_index is not None:
            order_index = int(order_index)
        if cycle is not None:
            cycle = int(cycle)

        record_trial(
            session_id=session_id,
            kind=kind,
            order_index=order_index,
            cycle=cycle,
            img=img,
            answer=answer,
            time_ms=time_ms,
        )
    except ValueError as e:
        return str(e), 400

    return jsonify({"ok": True})

@app.route("/api/finish", methods=["POST"])
def api_finish():
    data = request.get_json(silent=True) or {}
    session_id = str(data.get("session_id", "")).strip()
    if not session_id:
        return "Missing session_id", 400

    try:
        result = finish_session(session_id)
    except ValueError as e:
        return str(e), 400

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
    init_db()
    print(f"Serving on http://127.0.0.1:{PORT}")
    print(f"Images directory:   {IMGS_DIR}")
    print(f"Practice directory: {PRAC_DIR}")
    print(f"SQLite DB:          {DB_PATH}")
    print(f"CSV output:         {CSV_PATH}")
    app.run(host="0.0.0.0", port=PORT, debug=True)