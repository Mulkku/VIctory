# tools

## vrun.py: one reproducible game session and its verdict

Python 3, standard library only, Windows. Needs a diagnostics build
(`cargo build --profile diag -p victory --features diagnostics`) for the
software adapter, scripted input and frame captures.

Only one GPU-heavy job may run at a time, so launch it under the shared lock
(the bash builtin `flock`; vrun does not take the lock itself, it only
refuses to start while another vrun runs):

```sh
flock D:/victory-runs/gpu.lock python tools/vrun.py target/diag/victory.exe --label baseline
```

| Option | Default | Meaning |
|--------|---------|---------|
| `--label` | required | run name: folder suffix, ledger |
| `--mode warp\|hw` | `warp` | software adapter, or the real GPU (menu-only runs) |
| `--timeout S` | 1800 | wall seconds for the whole session |
| `--world-time S` | none | stop that long after the world stage is reached |
| `--hang S` | 600 | no log line and no frame for that long ends the run as a hang |
| `--script FILE` | `tools/first-boot.txt` | input script (below) |
| `--keys S`, `--pad S` | from the script file | `VICTORY_KEYS` / `VICTORY_PAD` given directly |
| `--settle N` | 30 | frames after a press its effect is looked for |
| `--stuck-keys N` | 3 | that many presses in a row without a visible effect end the run as stuck |
| `--still S` | 900 | short of the world, no capture changing for that long ends the run as stuck |
| `--data fresh\|DIR` | `fresh` | empty data folder, or a copy of a saved one (`DIR` is copied, never written) |
| `--cache DIR` | `D:\victory-runs\cache\<build>` | cache folder; shader entries are keyed by the build anyway |
| `--capture-every N` | 100 | capture every N-th presented frame too (0: powers of two only) |
| `--stages FILE` | `D:\victory-runs\stages.txt` | local stage rules (below) |
| `--host-input` | off | let host pads and the keyboard through |
| `--env K=V` | | extra environment, repeatable (e.g. `VICTORY_D3D12_DEBUG=1`) |

Each run gets `D:\victory-runs\<UTC time>-<label>\`:

- `victory.exe`: the copy that ran (the build can be replaced meanwhile);
- `victory.log`: stdout and stderr of every process of the session (a
  relaunch for a new game inherits them), each line prefixed with seconds
  since start;
- `data\`: the run's own `VICTORY_DATA_DIR` (saves, settings, `crashes.log`);
- `crashes.log`: a copy of the run's crash log entries;
- `dred.txt`: the host GPU failure report, with DRED data when the adapter
  gives it (`VICTORY_D3D12_DRED=1` is always set);
- `frames\`: `VICTORY_CAPTURE` output: powers of two, every N-th frame, and
  around each frame-timed press;
- `captures.json`: every capture's signature and difference to the one
  before;
- `run.json`: what ran (build hash, branch, commit, environment, scripts);
- `summary.json`: the verdict.

One line per run is appended to `D:\victory-runs\ledger.csv`
(timestamp, label, branch, commit, mode, stage, end, frames, seconds).

Everything under `D:\victory-runs` is local. Never commit or post it; report
counts only (docs/asset-policy.md).

### Evidence from the captures

Each capture is shrunk to a 32x18 grayscale signature. Two captures differ
in the cells whose gray level moved by more than 8; at most 2 such cells
count as near-identical (a loading spinner stays within that).

- A press had a visible effect when the captures just before it and
  `--settle` frames after it differ in more cells than the two captures
  before it (the screen's own motion). Frame-timed presses get those three
  captures; second-timed ones use the nearest captures.
- The run ends as `stuck` when the last `--stuck-keys` presses, made on a
  screen that is not black, had no visible effect and nothing changed since
  the first of them, or when (short of the world) no capture changed for
  `--still` seconds.
- `menus`: 64 frames presented and a capture that is not black. `loading`:
  its log rule. `world`: its log rule, and a capture after it that is not
  black and differs from each of the last three captures taken before the
  title left the menus (the first log rule).

### summary.json

- `stage` (`boot`, `menus`, `loading`, `world`), `stage_reached_at_s`,
  `stage_evidence`, `log_rules_at_s` (when each rule fired, confirmed or not);
- `end`: `exit` (the title ended; `exit_code`), `crash` (unhandled exception:
  `crash` holds the exception code, `module+offset` and detail),
  `device-removed` (host GPU failure: `device_removed_reason`, `dred` with
  DRED's stopped lists and the draws/dispatches in flight), `stuck`
  (`stuck_at`: why, the last distinct capture, the presses on that screen
  without effect), or `timeout`, `world-time`, `hang` (vrun stopped it:
  `stopped_by_harness` is `closed` when the window's close ended the title,
  else `terminated`);
- `seconds`, `seconds_in_world`, `frames_presented` (the log's power-of-two
  lines and the captures), `last_fps`, `black_frames_logged`;
- `captures`: count, black ones, the last, the last not black, and
  `near_identical_runs` (from/to frame and seconds, length; 3 or more);
- `presses`: per press the captures before and after, the difference
  (`score` mean gray difference, `changed_cells`), the screen's own motion
  (`ambient_cells`), `visible`, `on_black`;
- `untranslated_shaders`, `untranslated_by_error`: "shader does not
  translate" by stage and error kind;
- `dropped_draws`, `dropped_dispatches`: issued minus run on the host, summed
  over the log's frame lines (up to `dropped_counted_to_frame`).

### Input scripts

`VICTORY_KEYS=<when>:<key>,...` and `VICTORY_PAD=<when>:<button>,...`.
`<when>` is seconds after input starts (held 0.2 s) or `f<n>`, the n-th
presented frame (held 4 frames). The software adapter presents at a varying,
low rate, so frame times are what make a script repeatable. Keys: a letter,
a digit, `Enter`, `Esc`, `Tab`, `Back`, `Up`, `Down`, `Left`, `Right`.
Buttons: `A B X Y Menu View Up Down Left Right LB RB LS RS`, pressed on the
first pad the title sees (the first host pad, else the keyboard's stand-in).

vrun sets `VICTORY_KEYS_ONLY=1`: host pads and the host keyboard are left
out, so only the scripts reach the title, whatever is plugged in or pressed
meanwhile.

A script file has one press per line, `key <when>:<key>` or
`pad <when>:<button>`, `#` comments. `tools/first-boot.txt` takes a fresh
data folder into the world.

### Stage rules

`loading` and `world` are recognised by log lines that name the title's own
files, so the rules stay outside the repository, in
`D:\victory-runs\stages.txt`:

```
# STAGE REGEX (Python syntax), matched against each log record
loading <regex>
world   <regex>
```

### Stopping

At a timeout vrun sends the title's window the close command; VIctory then
suspends the title as the console does and ends the process. If that takes
over 30 s, vrun terminates the process.
