#!/usr/bin/env python3
"""Runs the title once, reproducibly, and leaves a verdict (tools/README.md).

Launch under the shared GPU lock:

    flock D:/victory-runs/gpu.lock python tools/vrun.py <victory.exe> --label <name>

Python 3 standard library only. Windows only (process control through ctypes).
"""

import argparse
import collections
import csv
import ctypes
import ctypes.wintypes as wt
import datetime
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
import zlib

RUNS = r"D:\victory-runs"
GAME_EXE = r"D:\gta\GTA6.Xbox.v1.0.2.1-CYBERLEEK\game_scarlett_master.exe"
DEVICE_REMOVED = 0x887A0005
# The window close VIctory turns into the console's suspend-and-end.
WM_SYSCOMMAND, SC_CLOSE = 0x0112, 0xF060

# Where a run is, from what the log says; the furthest stage seen wins.
STAGES = ["boot", "menus", "loading", "world"]
ANSI = re.compile(r"\x1b\[[0-9;]*m")
FIELD = re.compile(r"(\w+)=(\"(?:[^\"\\]|\\.)*\"|\S+)")
FPS = re.compile(r"\b(\d+(?:\.\d+)?) fps, ")
CRASH = re.compile(r"VIctory: unhandled exception (0x[0-9a-f]+) at (\S+)(?: \((.*)\))?")
RELAUNCHED = re.compile(r"\brelaunched\b.*\bpid=(\d+)")
# A host GPU failure report: DRED's stopped lists, and the draws and
# dispatches in flight.
STOPPED = re.compile(r"a list stopped at command (\d+) of (\d+) \(op (\d+)\)")
IN_FLIGHT = re.compile(r"list \d+ of \d+, of its \d+: (\w+)")

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32.OpenProcess.restype = wt.HANDLE
kernel32.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
kernel32.WaitForSingleObject.argtypes = [wt.HANDLE, wt.DWORD]
kernel32.GetExitCodeProcess.argtypes = [wt.HANDLE, ctypes.POINTER(wt.DWORD)]
kernel32.TerminateProcess.argtypes = [wt.HANDLE, wt.UINT]
kernel32.CloseHandle.argtypes = [wt.HANDLE]
ENUM = ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
user32.EnumWindows.argtypes = [ENUM, wt.LPARAM]
user32.GetWindowThreadProcessId.argtypes = [wt.HWND, ctypes.POINTER(wt.DWORD)]
user32.IsWindowVisible.argtypes = [wt.HWND]
user32.PostMessageW.argtypes = [wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM]


class Process:
    """A victory process by id: waited on, closed, ended."""

    def __init__(self, pid):
        self.pid = pid
        # SYNCHRONIZE | PROCESS_TERMINATE | PROCESS_QUERY_LIMITED_INFORMATION
        self.handle = kernel32.OpenProcess(0x00100000 | 0x0001 | 0x1000, False, pid)

    def exit_code(self):
        """The exit code, or None while it runs."""
        if not self.handle or kernel32.WaitForSingleObject(self.handle, 0) != 0:
            return None if self.handle else 0
        code = wt.DWORD()
        kernel32.GetExitCodeProcess(self.handle, ctypes.byref(code))
        return code.value

    def windows(self):
        found = []

        def each(hwnd, _):
            pid = wt.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            if pid.value == self.pid and user32.IsWindowVisible(hwnd):
                found.append(hwnd)
            return True

        user32.EnumWindows(ENUM(each), 0)
        return found

    def close(self):
        """Asks for the title's clean end (as the window's close button)."""
        hwnds = self.windows()
        for hwnd in hwnds:
            user32.PostMessageW(hwnd, WM_SYSCOMMAND, SC_CLOSE, 0)
        return bool(hwnds)

    def terminate(self):
        kernel32.TerminateProcess(self.handle, 1)


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def git(*args, cwd):
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else ""
    except OSError:
        return ""


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fields(line):
    return {k: v.strip('"') for k, v in FIELD.findall(line)}


KEY_NAMES = {"Enter", "Esc", "Tab", "Back", "Up", "Down", "Left", "Right"}
PAD_NAMES = {"A", "B", "X", "Y", "Menu", "View", "Up", "Down", "Left", "Right", "LB", "RB", "LS", "RS"}


def parse_script(script, names, letters=False):
    """The `VICTORY_KEYS`/`VICTORY_PAD` grammar (gameinput.rs `script`):
    `<when>:<name>` entries, `<when>` seconds or `f<n>` presented frames.
    Returns `(entry, "frame"|"seconds", value)`; exits on an entry the
    runtime would skip."""
    out = []
    for entry in filter(None, (e.strip() for e in script.split(","))):
        at, _, name = entry.partition(":")
        at, name = at.strip(), name.strip()
        if not (name in names or (letters and len(name) == 1 and name.isascii() and name.isalnum())):
            sys.exit(f"script: unknown name in {entry!r}")
        try:
            out.append((entry, "frame", int(at[1:])) if at.startswith("f")
                       else (entry, "seconds", float(at)))
        except ValueError:
            sys.exit(f"script: bad time in {entry!r}")
    return out


# Captures are compared by a signature: the frame shrunk to GRID cells of
# grayscale. Cells differing by more than CELL_DIFF count as changed; two
# captures with at most SAME_CELLS changed cells are near-identical (a
# loading spinner stays within that).
GRID = (32, 18)
CELL_DIFF = 8
SAME_CELLS = 2


def signature(path):
    """The capture as GRID cell means of grayscale (bytes), or None if it is
    not one of our RGB captures (filter 0 on every row)."""
    with open(path, "rb") as f:
        data = f.read()
    pos, width, height, idat = 8, 0, 0, []
    while pos + 8 <= len(data):
        (n,) = struct.unpack(">I", data[pos:pos + 4])
        kind, body = data[pos + 4:pos + 8], data[pos + 8:pos + 8 + n]
        if kind == b"IHDR":
            width, height = struct.unpack(">II", body[:8])
            if body[8:10] != b"\x08\x02":
                return None
        elif kind == b"IDAT":
            idat.append(body)
        pos += 12 + n
    raw = zlib.decompress(b"".join(idat))
    stride = width * 3 + 1
    gw, gh = GRID
    sums, counts = [0] * (gw * gh), [0] * (gw * gh)
    step = 3
    for y in range(0, height, step):
        row = raw[y * stride + 1:(y + 1) * stride]
        base = (y * gh // height) * gw
        for x in range(0, width, step):
            p = 3 * x
            cell = base + x * gw // width
            sums[cell] += row[p] * 3 + row[p + 1] * 6 + row[p + 2]
            counts[cell] += 10
    return bytes(s // max(c, 1) for s, c in zip(sums, counts))


def compare(a, b):
    """(mean absolute difference 0-255, cells changed) of two signatures."""
    diffs = [abs(x - y) for x, y in zip(a, b)]
    return round(sum(diffs) / len(diffs), 2), sum(d > CELL_DIFF for d in diffs)


class Captures:
    """Saved frames in order, their signatures and how each differs from the
    one before."""

    def __init__(self, folder):
        self.folder = folder
        self.list = []  # dicts: frame, t, sig, score, changed, black

    def add(self, frame, t):
        path = os.path.join(self.folder, f"frame-{frame}.png")
        try:
            sig = signature(path)
        except (OSError, ValueError, zlib.error):
            return False
        if sig is None:
            return True
        entry = {"frame": frame, "t": round(t, 1), "sig": sig, "black": max(sig) < 8}
        if self.list:
            entry["score"], entry["changed"] = compare(self.list[-1]["sig"], sig)
        else:
            entry["score"], entry["changed"] = None, None
        self.list.append(entry)
        return True

    def same(self, entry):
        return entry["changed"] is not None and entry["changed"] <= SAME_CELLS

    def last_distinct(self):
        """The first capture of the current run of near-identical ones."""
        i = len(self.list) - 1
        while i > 0 and self.same(self.list[i]):
            i -= 1
        return self.list[i] if self.list else None

    def runs(self, minimum=3):
        """Runs of at least `minimum` consecutive near-identical captures."""
        out, start = [], 0
        for i in range(1, len(self.list) + 1):
            if i < len(self.list) and self.same(self.list[i]):
                continue
            if i - start >= minimum:
                a, b = self.list[start], self.list[i - 1]
                out.append({"from_frame": a["frame"], "to_frame": b["frame"], "captures": i - start,
                            "from_s": a["t"], "to_s": b["t"], "black": a["black"]})
            start = i
        return out

    def before(self, frame=None, t=None):
        found = [c for c in self.list if (c["frame"] < frame if frame is not None else c["t"] < t)]
        return found[-1] if found else None

    def after(self, frame=None, t=None):
        for c in self.list:
            if (c["frame"] >= frame if frame is not None else c["t"] >= t):
                return c
        return None

    def table(self):
        return [{"frame": c["frame"], "t": c["t"], "signature": hashlib.sha1(c["sig"]).hexdigest()[:12],
                 "score": c["score"], "changed_cells": c["changed"], "black": c["black"]}
                for c in self.list]


class Run:
    def __init__(self, args):
        self.args = args
        self.dir = os.path.join(args.runs, f"{stamp()}-{args.label}")
        self.started = time.monotonic()
        self.stage = "boot"
        self.stage_at = {"boot": 0.0}
        self.evidence = {}
        self.log_stage_at = {}
        self.frames = 0
        self.fps = None
        self.fps_at = None
        self.black_frames = 0
        self.input_t = None
        self.saved = []  # (frame, t) of captures the log reported
        self.captures = None
        self.keys = []  # dicts: entry, kind, value, then the verdict
        self.last_output = time.monotonic()
        self.last_frame_change = time.monotonic()
        self.relaunches = []
        self.untranslated = collections.Counter()
        self.dropped_draws = 0
        self.dropped_dispatches = 0
        self.gpu_failure = None
        self.crash = None
        self.stuck_why = None
        self.lock = threading.Lock()

    def elapsed(self):
        return time.monotonic() - self.started

    def reach(self, stage, evidence):
        if STAGES.index(stage) > STAGES.index(self.stage):
            self.stage = stage
            self.stage_at[stage] = round(self.elapsed(), 1)
            self.evidence[stage] = evidence
            print(f"[vrun] {self.elapsed():7.1f}s stage {stage}: {evidence}", flush=True)

    def note(self, record):
        """Reads one log record (a line and its continuation lines)."""
        with self.lock:
            t = self.elapsed()
            self.last_output = time.monotonic()
            if "frame presented" in record:
                f = fields(record)
                n = int(f.get("frame", 0))
                if n > self.frames:
                    self.frames = n
                    self.last_frame_change = time.monotonic()
                self.dropped_draws += int(f.get("draws", 0)) - int(f.get("drawn", 0))
                self.dropped_dispatches += int(f.get("dispatches", 0)) - int(f.get("dispatched", 0))
            elif (m := FPS.search(record)) and "ms/frame" in record:
                self.fps = float(m.group(1))
                self.fps_at = round(t, 1)
                self.last_frame_change = time.monotonic()
            elif "frame saved" in record:
                if m := re.search(r"frame-(\d+)\.png", record):
                    n = int(m.group(1))
                    self.saved.append((n, t))
                    if n > self.frames:
                        self.frames = n
                        self.last_frame_change = time.monotonic()
            elif "shader does not translate" in record:
                f = fields(record)
                self.untranslated[f"{f.get('stage', '?')}: {error_kind(f.get('error', '?'))}"] += 1
            elif m := CRASH.search(record):
                self.crash = self.crash or {
                    "exception": m.group(1),
                    "at": m.group(2),
                    "detail": m.group(3) or "",
                }
            elif m := RELAUNCHED.search(record):
                self.relaunches.append(int(m.group(1)))
            elif "black_frames=" in record:
                self.black_frames = max(self.black_frames, int(fields(record)["black_frames"]))
            elif "GameInputCreate" in record and self.input_t is None:
                self.input_t = t
            if " ERROR victory_gpu::host:" in record and self.gpu_failure is None:
                self.gpu_failure = record
            for stage, pattern in self.args.stage_patterns:
                if stage not in self.log_stage_at and pattern.search(record):
                    self.log_stage_at[stage] = round(t, 1)
                    print(f"[vrun] {t:7.1f}s log rule for {stage}", flush=True)

    def progress(self):
        """Reads new captures, judges key presses and stages. Returns why the
        run is stuck, or None."""
        with self.lock:
            saved, self.saved = self.saved, []
        for i, (frame, t) in enumerate(saved):
            if not self.captures.add(frame, t):
                with self.lock:  # not readable yet: next time
                    self.saved = saved[i:] + self.saved
                break
        caps = self.captures
        if not caps.list:
            return None
        lit = next((c for c in caps.list if not c["black"]), None)
        if lit and self.frames >= self.args.menu_frames:
            self.reach("menus", f"{self.frames} frames presented; capture {lit['frame']} not black")
        if (t := self.log_stage_at.get("loading")) is not None:
            self.reach("loading", f"log rule at {t} s")
        if (t := self.log_stage_at.get("world")) is not None:
            # The last captures before the title left the menus.
            left = min(self.log_stage_at.values())
            menu = [c for c in caps.list if c["t"] < left][-3:]
            for c in caps.list:
                if c["t"] > t and not c["black"] and all(
                    compare(m["sig"], c["sig"])[1] > SAME_CELLS for m in menu
                ):
                    self.reach("world", f"log rule at {t} s; capture {c['frame']} differs from "
                                        f"the last {len(menu)} menu captures")
                    break
        self.judge_keys()
        return self.stuck()

    def judge_keys(self):
        """A key press had a visible effect when the captures before and after
        it differ more than the two captures before it (the screen's own
        motion)."""
        caps = self.captures
        settle = self.args.settle
        for k in self.keys:
            if "visible" in k:
                continue
            if k["kind"] == "frame":
                before = caps.before(frame=k["value"])
                after = caps.after(frame=k["value"] + settle)
            else:
                if self.input_t is None:
                    continue
                at = self.input_t + k["value"]
                before = caps.before(t=at)
                after = caps.after(t=at + settle / max(self.fps or 15, 1))
            if before is None or after is None:
                continue
            i = caps.list.index(before)
            pre = caps.list[i - 1] if i > 0 else None
            score, changed = compare(before["sig"], after["sig"])
            ambient = compare(pre["sig"], before["sig"])[1] if pre else 0
            k.update(before=before["frame"], after=after["frame"], score=score,
                     changed_cells=changed, ambient_cells=ambient,
                     visible=changed > max(SAME_CELLS, ambient + 1), on_black=before["black"])

    def stuck(self):
        """Stuck: the last `--stuck-keys` presses, on a screen that is not
        black (a black one is a transition), showed nothing and the screen
        has not changed since the first of them; or, short of the world, no
        capture changed for `--still` seconds."""
        caps, n = self.captures, self.args.stuck_keys
        judged = sorted((k for k in self.keys if "visible" in k), key=lambda k: k["before"])[-n:]
        since = caps.last_distinct()
        if n and len(judged) == n and not any(k["visible"] or k["on_black"] for k in judged):
            if since["frame"] <= judged[0]["before"]:
                return f"the last {n} presses had no visible effect"
        if self.stage != "world" and caps.list[-1]["t"] - since["t"] >= self.args.still:
            return f"no capture changed for {self.args.still:.0f} s"
        return None


def error_kind(error):
    """A translation error without where it happened (code offsets)."""
    kind = re.sub(r"^.*?code at 0x[0-9a-f]+: ", "", error)
    return re.sub(r"0x[0-9a-f]+|\b\d+\b", "#", kind)


def load_stages(path):
    """`STAGE<whitespace>REGEX` per line, `#` comments: a log record matching
    REGEX means STAGE is reached. Kept outside the repository (the rules
    name the title's own files)."""
    rules = []
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                stage, pattern = line.split(None, 1)
                if stage not in STAGES:
                    sys.exit(f"{path}: {stage} is none of {STAGES}")
                rules.append((stage, re.compile(pattern)))
    return rules


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("victory", help="path to victory.exe (a diagnostics build for warp)")
    p.add_argument("--label", required=True, help="run name (folder suffix, ledger)")
    p.add_argument("--mode", choices=["warp", "hw"], default="warp",
                   help="warp: software adapter (default); hw: real GPU, menu-only runs")
    p.add_argument("--timeout", type=float, default=1800, help="wall seconds for the session")
    p.add_argument("--world-time", type=float, default=None,
                   help="once in the world, stop after this many seconds there")
    p.add_argument("--hang", type=float, default=600,
                   help="no frame and no log line for this long counts as a hang")
    p.add_argument("--keys", default=None,
                   help="VICTORY_KEYS script (default: the key lines of tools/first-boot.txt)")
    p.add_argument("--pad", default=None,
                   help="VICTORY_PAD script (default: the pad lines of tools/first-boot.txt)")
    p.add_argument("--script", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                    "first-boot.txt"),
                   help="script file used when --keys/--pad are not given")
    p.add_argument("--settle", type=int, default=30,
                   help="presented frames after a press its effect is looked for")
    p.add_argument("--stuck-keys", type=int, default=3,
                   help="this many presses in a row without a visible effect end the run as stuck")
    p.add_argument("--still", type=float, default=900,
                   help="short of the world, no capture changing for this long ends the run as stuck")
    p.add_argument("--data", default="fresh",
                   help="'fresh' (empty data folder) or a snapshot folder copied in")
    p.add_argument("--cache", default=None,
                   help="cache folder (default: per victory.exe build, under the runs folder)")
    p.add_argument("--game", default=GAME_EXE, help="the title's executable")
    p.add_argument("--capture-every", type=int, default=100,
                   help="also save every n-th presented frame (0: powers of two only)")
    p.add_argument("--capture-max", type=int, default=20000)
    p.add_argument("--menu-frames", type=int, default=64,
                   help="presented frames that count as reaching the menus")
    p.add_argument("--stages", default=os.path.join(RUNS, "stages.txt"),
                   help="local stage rules (STAGE REGEX per line; see tools/README.md)")
    p.add_argument("--host-input", action="store_true",
                   help="let host pads and the keyboard through (default: scripted keys only)")
    p.add_argument("--env", action="append", default=[], help="extra KEY=VALUE")
    p.add_argument("--runs", default=RUNS, help="runs folder")
    args = p.parse_args()
    args.stage_patterns = load_stages(args.stages)

    victory = os.path.abspath(args.victory)
    if not os.path.isfile(victory):
        sys.exit(f"no {victory}")
    if args.keys is None or args.pad is None:
        script = {"key": [], "pad": []}
        with open(args.script, encoding="utf-8") as f:
            for line in f:
                line = line.split("#", 1)[0].strip()
                if line:
                    kind, _, entry = line.partition(" ")
                    if kind not in script:
                        sys.exit(f"{args.script}: lines start with 'key' or 'pad': {line!r}")
                    script[kind].append(entry.strip())
        args.keys = ",".join(script["key"]) if args.keys is None else args.keys
        args.pad = ",".join(script["pad"]) if args.pad is None else args.pad
    presses = [("key", *k) for k in parse_script(args.keys, KEY_NAMES, letters=True)]
    presses += [("pad", *k) for k in parse_script(args.pad, PAD_NAMES)]

    run = Run(args)
    run.keys = [{"device": d, "entry": e, "kind": k, "value": v} for d, e, k, v in presses]
    marker = guard(args.runs, run.dir)
    try:
        os.makedirs(run.dir)
        session(run, victory)
    finally:
        os.remove(marker)


def guard(runs, run_dir):
    """A second line of defence besides flock: no two harnesses at once."""
    marker = os.path.join(runs, "vrun.running")
    if os.path.exists(marker):
        with open(marker, encoding="utf-8") as f:
            pid, _, other = f.read().partition("\n")
        if pid.isdigit() and Process(int(pid)).exit_code() is None and int(pid) != os.getpid():
            sys.exit(f"another vrun (pid {pid}, {other}) is running; use the GPU lock")
    with open(marker, "w", encoding="utf-8") as f:
        f.write(f"{os.getpid()}\n{run_dir}")
    return marker


def session(run, victory):
    args = run.args
    d = run.dir
    data = os.path.join(d, "data")
    if args.data == "fresh":
        os.makedirs(data)
    else:
        shutil.copytree(args.data, data)
    build = sha256(victory)
    cache = args.cache or os.path.join(args.runs, "cache", build[:16])
    os.makedirs(cache, exist_ok=True)
    frames_dir = os.path.join(d, "frames")
    os.makedirs(frames_dir)
    run.captures = Captures(frames_dir)
    # The run uses its own copy, so the build can be replaced meanwhile.
    copy = os.path.join(d, "victory.exe")
    shutil.copy2(victory, copy)

    env = dict(os.environ)
    env.update(
        VICTORY_GAME_DIR=os.path.dirname(args.game),
        VICTORY_DATA_DIR=data,
        VICTORY_CACHE_DIR=cache,
        VICTORY_CAPTURE=frames_dir,
        VICTORY_D3D12_DRED="1",
        VICTORY_KEYS=args.keys,
        VICTORY_PAD=args.pad,
        NO_COLOR="1",
    )
    if not args.host_input:
        env["VICTORY_KEYS_ONLY"] = "1"
    # Every n-th frame, and around each frame-timed press: two before it (the
    # screen's own motion) and one `settle` frames after it.
    wanted = set(range(args.capture_every, args.capture_max + 1, args.capture_every)
                 if args.capture_every else ())
    for k in run.keys:
        if k["kind"] == "frame":
            n = k["value"]
            wanted |= {max(n - 1 - args.settle, 1), max(n - 1, 1), n + args.settle}
    if wanted:
        env["VICTORY_CAPTURE_FRAMES"] = ",".join(str(n) for n in sorted(wanted))
    if args.mode == "warp":
        env["VICTORY_D3D12_WARP"] = "1"
    else:
        env.pop("VICTORY_D3D12_WARP", None)
    for kv in args.env:
        k, _, v = kv.partition("=")
        env[k] = v

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vdir = os.path.dirname(victory)
    src = git("rev-parse", "--show-toplevel", cwd=vdir) or repo
    meta = {
        "label": args.label,
        "victory": victory,
        "victory_sha256": build,
        "branch": git("rev-parse", "--abbrev-ref", "HEAD", cwd=src),
        "commit": git("rev-parse", "--short", "HEAD", cwd=src),
        "dirty": bool(git("status", "--porcelain", "--untracked-files=no", cwd=src)),
        "mode": args.mode,
        "data": args.data,
        "cache": cache,
        "keys": args.keys,
        "pad": args.pad,
        "timeout": args.timeout,
        "world_time": args.world_time,
        "env": {k: env[k] for k in sorted(env) if k.startswith("VICTORY_") and k != "VICTORY_CAPTURE_FRAMES"},
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    with open(os.path.join(d, "run.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    command = [copy, "run", args.game, "--execute"]
    print(f"[vrun] {d}\n[vrun] {' '.join(command)}", flush=True)
    log = open(os.path.join(d, "victory.log"), "w", encoding="utf-8", errors="replace")
    child = subprocess.Popen(
        command, cwd=d, env=env, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    proc = Process(child.pid)

    def pump():
        # Both streams, every process of the session (a relaunch inherits
        # them). A record is a line plus its continuation lines (indented
        # or starting with ')', as in multi-line failure reports).
        record = None
        for raw in iter(child.stdout.readline, b""):
            line = ANSI.sub("", raw.decode("utf-8", "replace").rstrip("\r\n"))
            log.write(f"[{run.elapsed():8.1f}] {line}\n")
            log.flush()
            if record is not None and line[:1] in (" ", "\t", ")"):
                record += "\n" + line
                continue
            if record is not None:
                run.note(record)
            record = line
        if record is not None:
            run.note(record)

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()

    end, code = None, None
    world_since = None
    followed = set()
    while True:
        time.sleep(1)
        code = proc.exit_code()
        if code is not None:
            # A relaunch (new game) replaces the process: follow the new one
            # (its log line may trail the old process's end a little).
            time.sleep(3)
            with run.lock:
                pending = [pid for pid in run.relaunches if pid not in followed]
            if pending:
                followed.add(pending[0])
                proc = Process(pending[0])
                print(f"[vrun] {run.elapsed():7.1f}s relaunched as {proc.pid}", flush=True)
                continue
            end = "exit"
            break
        if run.stage == "world" and world_since is None:
            world_since = time.monotonic()
        quiet = time.monotonic() - max(run.last_output, run.last_frame_change)
        why = run.progress()
        if why:
            run.stuck_why = why
            end = "stuck"
        elif args.world_time is not None and world_since and time.monotonic() - world_since >= args.world_time:
            end = "world-time"
        elif run.elapsed() >= args.timeout:
            end = "timeout"
        elif quiet >= args.hang:
            end = "hang"
        if end:
            break

    stopped = None
    if end != "exit":
        stopped = stop(proc)
        code = proc.exit_code()
    reader.join(timeout=30)
    log.close()
    finish(run, meta, end, code, stopped)


def stop(proc):
    """Clean title end if the window takes it, else terminate."""
    if proc.close():
        for _ in range(30):
            time.sleep(1)
            if proc.exit_code() is not None:
                return "closed"
    proc.terminate()
    for _ in range(10):
        time.sleep(1)
        if proc.exit_code() is not None:
            break
    return "terminated"


def finish(run, meta, end, code, stopped):
    args = run.args
    d = run.dir
    crashes = [
        os.path.join(root, "crashes.log")
        for root, _, files in os.walk(os.path.join(d, "data"))
        if "crashes.log" in files
    ]
    crash_text = ""
    for c in crashes:
        with open(c, encoding="utf-8", errors="replace") as f:
            crash_text += f.read()
    if crash_text:
        with open(os.path.join(d, "crashes.log"), "w", encoding="utf-8") as f:
            f.write(crash_text)
    if run.gpu_failure:
        with open(os.path.join(d, "dred.txt"), "w", encoding="utf-8") as f:
            f.write(run.gpu_failure + "\n")
    if run.crash is None and (m := CRASH.search(crash_text)):
        run.crash = {"exception": m.group(1), "at": m.group(2), "detail": m.group(3) or ""}

    if code is not None and code >= 0x80000000:
        code_text = f"{code:#010x}"
    else:
        code_text = str(code)
    if end == "exit":
        if code == DEVICE_REMOVED or run.gpu_failure:
            how = "device-removed"
        elif run.crash or (code is not None and code >= 0xC0000000):
            how = "crash"
        else:
            how = "exit"
    else:
        how = end
    failure = run.gpu_failure or ""
    reason = None
    if m := re.search(r"device removed: (.*?)(?:;|$)", failure):
        reason = m.group(1).strip()
    stopped_ops = collections.Counter(int(m.group(3)) for m in STOPPED.finditer(failure))
    in_flight = collections.Counter(m.group(1) for m in IN_FLIGHT.finditer(failure))
    run.progress()  # captures saved after the last look
    caps = run.captures
    frames = max([run.frames] + [c["frame"] for c in caps.list])
    lit = [c["frame"] for c in caps.list if not c["black"]]
    distinct = caps.last_distinct()
    with open(os.path.join(d, "captures.json"), "w", encoding="utf-8") as f:
        json.dump(caps.table(), f, indent=1)
    summary = {
        "label": args.label,
        "run_dir": d,
        "branch": meta["branch"],
        "commit": meta["commit"],
        "dirty": meta["dirty"],
        "mode": args.mode,
        "stage": run.stage,
        "stage_reached_at_s": run.stage_at,
        "stage_evidence": run.evidence,
        "log_rules_at_s": run.log_stage_at,
        "end": how,
        "stuck_at": {
            "why": run.stuck_why,
            "last_distinct_capture": distinct["frame"],
            "file": f"frames/frame-{distinct['frame']}.png",
            "at_s": distinct["t"],
            "presses_without_effect": [k["device"] + " " + k["entry"] for k in run.keys
                                       if k.get("visible") is False and k["before"] >= distinct["frame"]],
        } if how == "stuck" else None,
        "stopped_by_harness": stopped,
        "exit_code": code_text,
        "seconds": round(run.elapsed(), 1),
        "seconds_in_world": round(run.elapsed() - run.stage_at["world"], 1) if "world" in run.stage_at else 0,
        "frames_presented": frames,
        "last_fps": run.fps,
        "last_fps_at_s": run.fps_at,
        "black_frames_logged": run.black_frames,
        "captures": {
            "count": len(caps.list),
            "black": len(caps.list) - len(lit),
            "last": caps.list[-1]["frame"] if caps.list else None,
            "last_not_black": lit[-1] if lit else None,
            "near_identical_runs": caps.runs(),
        },
        "presses": [
            {k2: k[k2] for k2 in ("device", "entry", "before", "after", "score", "changed_cells",
                                  "ambient_cells", "visible", "on_black") if k2 in k}
            for k in run.keys
        ],
        "relaunches": len(run.relaunches),
        "crash": run.crash,
        "crash_log_entries": crash_text.count("unix time "),
        "device_removed_reason": reason,
        "dred": {
            "lists_stopped": sum(stopped_ops.values()),
            "stopped_at_op": {str(k): v for k, v in sorted(stopped_ops.items())},
            "page_fault": "page fault at" in failure,
            "in_flight": dict(in_flight.most_common()),
        } if failure else None,
        "untranslated_shaders": sum(run.untranslated.values()),
        "untranslated_by_error": dict(run.untranslated.most_common()),
        "dropped_draws": run.dropped_draws,
        "dropped_dispatches": run.dropped_dispatches,
        "dropped_counted_to_frame": run.frames,
    }
    with open(os.path.join(d, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    ledger = os.path.join(args.runs, "ledger.csv")
    new = not os.path.exists(ledger)
    with open(ledger, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "label", "branch", "commit", "mode", "stage", "end", "frames", "seconds"])
        w.writerow([os.path.basename(d)[:16], args.label, meta["branch"], meta["commit"],
                    args.mode, run.stage, how, frames, round(run.elapsed())])
    print(f"[vrun] {how} at stage {run.stage}, {frames} frames, {run.elapsed():.0f} s: "
          f"{os.path.join(d, 'summary.json')}", flush=True)


if __name__ == "__main__":
    main()
