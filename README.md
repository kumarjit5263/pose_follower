# pose_follower

Follows the first person who comes into view, draws their pose diagram (the
"stick figure") on them, and lets them control it with two-handed gestures:
stop following, resume following (locking onto whoever made the gesture), and
show / hide the pose diagram.

It uses the live camera when one is available; otherwise it plays the recorded
sessions in `tests/scenarios/`. Live sessions can be recorded (`--record`) and
replayed later with exactly the same results, which is how the test scenarios
are made.

- [Quick start](#quick-start)
- [Gestures](#gestures)
- [What you see on screen](#what-you-see-on-screen)
- [Running](#running)
- [Recording and replaying sessions](#recording-and-replaying-sessions)
- [Output files](#output-files)
- [Settings](#settings)
- [How it works](#how-it-works)
- [Code layout](#code-layout)
- [Tests](#tests)
- [Limitations](#limitations)
- [Troubleshooting](#troubleshooting)

## Quick start

Requires Python 3.12 (Linux or macOS; the setup script is bash).

```bash
./setup_env.sh              # create .venv and install everything (safe to re-run)
source .venv/bin/activate
python follow_pose.py       # live camera if present, else the recordings in tests/scenarios/
python -m pytest            # run the test suite
```

## Gestures

Each gesture is **two hands, one showing each hand shape**, held for **3
seconds**. Which gestures are available depends on whether you're being followed:

| Hold... | ...to | Available |
|---|---|---|
| open palm + **fist** | **STOP** following | while following |
| open palm + **scissors** | **RESUME** following - and lock onto the person who made the gesture, even if someone else is more prominent | while stopped (from anyone in view) |
| **fist** + **scissors** | **show / hide** the pose diagram on the body; tracking keeps running either way | while following |

The hand shapes:

| Shape | Fingers | Thumb |
|---|---|---|
| open palm | all four straight | out |
| fist | all four curled | tucked in |
| scissors | index + middle straight, ring + pinky curled | tucked in |

Anything else - e.g. a V-sign with the thumb out, a thumbs-up, an open hand
with the thumb folded in, three fingers - is read as *none* and never triggers
anything. Two identical shapes (two fists, two palms, ...) aren't a gesture.

Rules of the hold:

- **3 s hold.** A progress bar shows it. It starts counting when both shapes
  first appear, but is only shown (and logged) once they've been seen together
  for 0.4 s, so momentary mis-detections while your hands change shape never
  start one.
- **Short drop-outs are tolerated.** If the detector loses the hands for up to
  0.3 s the hold continues; longer, and it starts again.
- **Lower your hands afterwards.** After a gesture fires it's ignored until it
  has been gone for 0.5 s, so holding it longer never repeats it. This matters
  for the pose-diagram toggle (following stays on, so the same gesture stays
  available). Stop and resume can't repeat anyway: once stopped only resume is
  listened for, and vice versa.
- **Only the followed person's hands count** while following - a bystander's
  gesture is ignored. While stopped, anyone's hands count (so anyone can resume).
- **Resuming doesn't change the pose diagram** - if it was hidden, it stays hidden.

## What you see on screen

| Where | What |
|---|---|
| Top (white) | **Header**: each gesture you can do right now and what it will cause, e.g. `Hold open palm + fist to STOP tracking` / `Hold fist + scissors to HIDE the body pose drawing`, or `Hold open palm + scissors to RESUME tracking (on whoever shows it)` while stopped. It updates as things change (e.g. *HIDE* becomes *SHOW*). |
| Below the header | **Status**: `FOLLOWING` (green), `FOLLOWING (searching for person...)` (green, nobody detected right now) or `STOPPED` (red). |
| Below the status | **Banner** (yellow, 1.5 s) after something changes: `FOLLOWING STOPPED`, `FOLLOWING RESUMED`, `POSE DRAWING ON` / `OFF`. |
| On the person | The **pose diagram** (green joints, cyan bones) - only for the followed person, only while following, and only if not hidden. |
| Under each hand | The recognised shape (`PALM`, `FIST`, `SCISSORS`, `NONE`), for the hands that count. |
| Bottom | **Hold progress** (`HOLD open palm + fist to stop: 1.6 / 3 s` and a bar), or `Lower your hands` after a gesture fired. |
| Dashed yellow box | `locking on...` - right after a resume, where the app is looking for the person who made the gesture. |

When replaying a landmark recording (no video), the frame is a blank canvas
and the hand skeletons are drawn too, so the gestures are visible.

## Running

```bash
python follow_pose.py [options]
```

| Option | Meaning |
|---|---|
| *(none)* | Use the live camera if one delivers frames; otherwise play every recording in `tests/scenarios/`. |
| `--camera INDEX` | Camera to try (default `0`). |
| `--video PATH ...` | Skip the camera; play these videos, `.landmarks.jsonl` recordings and/or folders. |
| `--scenarios DIR` | Folder to play when there's no camera (default `tests/scenarios/`). |
| `--record [NAME]` | Record a live camera session (see below). |
| `--duration SECONDS` | Stop after this many seconds of footage. |
| `--outputs DIR` | Where output files go (default `outputs/`). |
| `--annotate` | Also save a video of what the window shows. |
| `--no-save` | Don't write output files when playing recordings. |
| `--no-display` | No window. Automatic when there's no display (e.g. over SSH); events are printed instead. |

Keys: `q` or `Esc` quits. In a window, recordings play at real-time speed;
headless, as fast as possible. Either way their timing (e.g. the 3 s hold) uses
the recording's own timestamps, so results are the same however fast the
computer is.

## Recording and replaying sessions

```bash
python follow_pose.py --record my_session     # press q (or Ctrl-C headless) to stop
```

This runs the app on the camera as normal and saves into `outputs/my_session/`:

| File | Contents |
|---|---|
| `my_session.mp4` | The raw camera footage (mirrored, as processed), re-timed to the real capture rate so a 3 s gesture lasts 3 s on playback. |
| `my_session.landmarks.jsonl` | The MediaPipe output for every frame: pose + hand landmarks. No images. |
| `my_session.actions.txt` | The timeline of what happened (see [Output files](#output-files)). |
| `my_session_annotated.mp4` | Only with `--annotate`: what the window showed. |

An existing folder is never overwritten. To turn a recording into a test
scenario, copy `my_session.landmarks.jsonl` into `tests/scenarios/` (and
optionally `my_session.actions.txt` as `my_session.live_actions.txt`, and a
`my_session.expected.json`) - see [tests/scenarios/README.md](tests/scenarios/README.md).

**Replays are exact.** Replaying a `.landmarks.jsonl` file (e.g.
`python follow_pose.py --video outputs/my_session/my_session.landmarks.jsonl`)
feeds the recorded MediaPipe output through the same logic with the same
timestamps, so it makes exactly the same decisions as the live session did -
every action, to the millisecond. No video or MediaPipe is needed.

## Output files

Each run on a recording (video or `.landmarks.jsonl`) writes into
`outputs/<name>/` (live sessions only with `--record`):

- `<name>.actions.txt` - the **action log**: a header (source, frame size, the
  gesture rules), a timeline, and a summary (duration, % of frames the person
  was followed / stopped, number of stops, resumes and drawing toggles).
  Replays of a landmark file are named `<name>.replay.actions.txt`.
- `<name>.landmarks.jsonl` - for video inputs, the MediaPipe output extracted
  from it (replayable like a recording).
- `<name>_annotated.mp4` - with `--annotate`.

Actions in the log:

| Action | Meaning |
|---|---|
| `START` / `END` | Session start / end (with totals). |
| `PERSON_FOUND` / `PERSON_LOST` | The followed person came into / left view (ignoring blips under 0.3 s; stamped when the change began). |
| `GESTURE_STARTED` | A gesture hold began (after 0.4 s), saying what it would do. |
| `GESTURE_TOO_SHORT` | A hold ended before 3 s - nothing changed. |
| `FOLLOWING_STOPPED` / `FOLLOWING_RESUMED` | Following stopped / resumed. |
| `POSE_DIAGRAM_HIDDEN` / `POSE_DIAGRAM_SHOWN` | The pose diagram was hidden / shown. |
| `HANDS_LOWERED` | After the pose-diagram toggle, the hands went down, so it can be used again. |

**`.landmarks.jsonl` format** (JSON Lines): the first line is a header -
`{"format": "pose_follower.landmarks", "version": 1, "source", "kind",
"width", "height", "fps", "mirrored", "mediapipe", "created", ...}` (`fps` is
the measured rate; a camera's reported rate is kept as `nominal_fps` if
different). Every further line is one frame: `{"i": index, "t": seconds,
"pose": [[x, y, z, visibility] x 33] or null, "hands": [[[x, y, z] x 21], ...]}`,
in MediaPipe's normalised image coordinates. Values are stored so that they
read back bit-identical to MediaPipe's, which is what makes replays exact.

`outputs/` is git-ignored (recordings contain faces).

## Settings

All tunables are in `follower/logic/config.py` (`Settings`):

| Setting | Default | Meaning |
|---|---|---|
| `stop_gesture` | `("PALM", "FIST")` | Stops following (while following). |
| `resume_gesture` | `("PALM", "SCISSORS")` | Resumes following (while stopped). |
| `draw_toggle_gesture` | `("FIST", "SCISSORS")` | Shows / hides the pose diagram (while following). |
| `hold_seconds` | `3.0` | How long a gesture must be held. |
| `hold_confirm` | `0.4` | How long it must be seen before the hold is shown / logged. |
| `release_seconds` | `0.5` | How long it must be gone ("hands down") before it's accepted again. |
| `dropout_grace` | `0.3` | Drop-out tolerated during a hold without restarting it. |
| `label_confirm_frames` / `label_confirm_seconds` | `2` / `0.1` | A hand's shown shape only changes once the new one is seen on this many frames over this long (no flicker). |
| `track_timeout` | `0.5` | How long a briefly lost hand keeps its identity. |
| `hand_near_body` | `3.0` | While following, a hand counts only within this many shoulder widths of the person's wrist. |
| `shoulders_per_hand` | `4.0` | Body-size estimate from hand size, used when the shoulders aren't visible. |
| `min_visibility` | `0.5` | Pose landmark visibility needed to use it. |
| `flash_seconds` | `1.5` | How long the banner shows. |
| `event_debounce` | `0.3` | A person must be found / lost this long before it's logged. |

Hand-shape thresholds (measured on recorded live sessions) are constants in
`follower/logic/gestures.py`: a finger is straight below 45° of bend and curled
above 80°; the thumb is tucked when its tip is within 0.35 hand-sizes of the
fingers' joints. MediaPipe confidence levels are arguments of
`MediaPipeDetector` in `follower/vision/detector.py`.

## How it works

Per frame:

1. **Input** (`follower/vision/capture.py`) - a camera frame (mirrored, stamped
   with session time), a video frame (stamped with its position in the video),
   or a frame of a landmark recording.
2. **Detection** (`follower/vision/detector.py`) - MediaPipe Pose (one person,
   tracked from frame to frame) and MediaPipe Hands (up to 4 hands, so one
   person's hands can't hide another's gesture). Pose runs on every frame, also
   while stopped, so recordings are complete. Two additions:
   - *Missed hands are recovered from the pose.* MediaPipe Hands often misses
     a closed hand or a smallish one. When the pose shows a raised wrist with no
     hand detected near it, a square around that wrist is cropped, enlarged and
     run through the hand model again (~20 ms, only while a hand is missing).
   - *Locking onto the person who resumed.* Until that person is found, pose
     detection sees only the band of the image around their gesturing hands
     (the rest is greyed out); after that, normal tracking follows them.
3. **Hand shapes** (`follower/logic/gestures.py`) - each hand is classified
   from 3-D finger-bend angles and the thumb's position (rotation-, scale- and
   aspect-ratio-independent); duplicate detections of one hand are merged; each
   hand is tracked between frames and its label smoothed (a single odd frame
   never changes it, and a stale label never counts towards a gesture).
4. **Decisions** (`follower/logic/tracking.py`) - which hands belong to the
   followed person; three hold-timers (stop / resume / show-hide), only the
   ones that make sense in the current state being active; following and
   pose-diagram state; who to lock onto after a resume.
5. **Reporting** - the action log (`follower/logic/actions.py`), drawing
   (`follower/vision/drawing.py`) and output files (`follower/vision/outputs.py`).

Steps 3-4 and the action log are pure Python with no cv2 / mediapipe, which is
why recordings can be replayed and unit-tested anywhere.

## Code layout

| Path | What | Imports cv2 / mediapipe? |
|---|---|---|
| `follow_pose.py` | Entry point. | - |
| `follower/logic/` | Settings, hand-shape classification, hand tracking & smoothing, gesture hold-timers, follow state, action log, landmark-file format, finding recordings. | **No** (enforced by `tests/test_layering.py`) |
| `follower/vision/` | Camera / video / replay input, MediaPipe inference, drawing, output files, main loop and CLI. | Yes |
| `tests/` | Test-only code: unit tests, synthetic landmarks and scenarios, the test-case generator. | Some |
| `tests/scenarios/` | Recorded live sessions used as app input and regression tests ([README](tests/scenarios/README.md)). | - |
| `setup_env.sh`, `requirements*.txt`, `requirements.lock` | Environment setup (see below). | - |
| `outputs/` | Generated files (git-ignored). | - |

### Environment

`./setup_env.sh` finds Python 3.12 (or installs it with `uv`, if available),
creates `.venv` (rebuilding it if it's missing, broken or the wrong version),
installs the exact versions in `requirements.lock` (falling back to the ranges
in the requirements files if those can't be installed), and checks the
imports. Options: `--recreate` (rebuild from scratch), `--no-lock`,
`--runtime-only` (skip test dependencies), `PYTHON=/path/to/python3.12`.

- `requirements.txt` - runtime: `mediapipe<0.10.30` (0.10.30 removed the
  `mp.solutions` API this app uses; on Python 3.12 this is 0.10.21),
  `opencv-contrib-python` (the OpenCV build mediapipe itself uses - installing
  `opencv-python` as well would clash), `numpy`.
- `requirements-dev.txt` - adds `pytest` and `scikit-image` (for the public-domain
  photo used in the real-MediaPipe test cases).
- `requirements.lock` - the known-good versions; regenerate with
  `./setup_env.sh --no-lock && .venv/bin/pip freeze` after changing requirements.

## Tests

```bash
python -m pytest                                   # everything (~3 min)
python -m pytest -m "not vision and not scenario"  # pure logic only (~5 s)
python -m pytest -m scenario                       # the recordings in tests/scenarios/ (~1 min)
python -m tests.generate_cases                     # write every test case + its outputs to outputs/test_cases/
```

What's covered:

- **Pure logic** - hand-shape classification (rotation, scale, aspect ratio,
  fingers pointing at the camera, thumb rules), smoothing and duplicate
  handling, the hold-timers, which hands count, the follow / stop / resume /
  show-hide state machine, the action log, the landmark-file format, and that
  `follower/logic` never imports cv2 / mediapipe.
- **Synthetic scenarios** (`tests/cases.py`) - 13 scripted sessions (e.g. stop
  then another person resumes, too-short holds, drop-outs, bystanders,
  flickering labels, holding the show/hide gesture for 7 s), checked through
  the logic and replayed through the full app.
- **Vision** - capture, MediaPipe wrapper, hand recovery, locking onto the
  gesturer (real MediaPipe on two people), drawing, output files, recording
  with a simulated camera, and the CLI.
- **Recorded live sessions** (`tests/scenarios/`) - each replayed through the
  full app and checked against its `.expected.json`, and against the action
  log written live while it was recorded (must match exactly).

`python -m tests.generate_cases` writes each case's input, expected result and
the app's outputs (action log, annotated video) to `outputs/test_cases/`, with
a `SUMMARY.md` of expected vs. detected results.

## Limitations

- **One person at a time.** MediaPipe's Pose model tracks a single person. If
  the followed person leaves, whoever is detected next is followed.
- **Locking onto the gesturer is best-effort.** It finds the person whose body
  is in the band around the gesturing hands; if someone else stands in that
  band it may pick them, and if nobody is found there within 30 frames (1 s at
  30 fps, ~3 s at 10 fps) it falls back to whoever is detected.
- **Hand ownership is by distance.** A bystander standing right next to the
  followed person (within 3 shoulder widths of their wrists) can still stop it.
- **While stopped, two people's hands can combine** (e.g. one shows a palm, the
  other scissors) to resume.
- **Speed.** Pose + hands run on the CPU; live sessions have been recorded at
  ~8-10 fps. Recovering a missed hand costs ~20 ms per hand while it's missing.
  Timing stays correct at any frame rate.
- **Setup script is bash** (Linux / macOS). On Windows, create a Python 3.12
  venv and `pip install -r requirements-dev.txt -c requirements.lock` by hand.

## Troubleshooting

- **"No camera available ... looking for recorded scenarios"** - no camera
  delivered a frame at that index; try `--camera 1`, or check permissions.
- **No window appears** - without a display (e.g. SSH) the app runs headless
  and prints events; use `--annotate` to save what the window would show.
- **`ImportError: libGL.so.1`** (Linux) - install OpenCV's system libraries,
  e.g. `sudo apt install libgl1 libglib2.0-0`.
- **`module 'mediapipe' has no attribute 'solutions'`** - mediapipe 0.10.30 or
  newer is installed; re-run `./setup_env.sh --recreate`.
- **A scenario test says the replay differs from the live session** - the
  gestures or their settings changed since it was recorded; re-record it.
