# Recorded test scenarios

Recordings in this folder are used in two ways:

1. **As the app's input when no camera is available.** `python follow_pose.py`
   tries the camera first; if there is none it plays everything here and writes
   `outputs/<name>/<name>.replay.actions.txt` (the action log) for each.
2. **As regression tests.** `python -m pytest -m scenario` replays each one
   through the full app and checks it (see [Checks](#checks)).

## Included recordings

Live webcam sessions (1280x720, ~8-9 fps), recorded with `--record` using the
current gestures (open palm + fist = stop, open palm + scissors = resume,
fist + scissors = show / hide the pose diagram). Landmarks only - no images.

| Recording | Length | What happens | Result |
|---|---|---|---|
| `default_check` | ~42 s | Each gesture once, in the normal flow: hide the pose diagram (~9.9 s), show it again (~16.7 s), stop (~22.8 s), resume (~38.1 s). | stop, resume; 2 drawing toggles |
| `detection_check` | ~56 s | Hand shapes that must **not** trigger anything: two fists, two scissors, two open palms, a palm with another shape, other shapes, and one fist + scissors held only ~1.4 s. | nothing changes; followed throughout |
| `toggle_check` | ~80 s | Repeated attempts: too-short holds of each gesture, then hide the pose diagram (~20.7 s), stop (~46.5 s), a too-short resume, and resume (~74.4 s); the diagram stays hidden after resuming; the person then walks out of view. | stop, resume; 1 drawing toggle |

Each comes with three files:

| File | What |
|---|---|
| `<name>.landmarks.jsonl` | The MediaPipe output for every frame, as recorded live. |
| `<name>.live_actions.txt` | The action log the app wrote **live** while recording. |
| `<name>.expected.json` | The expected result (see below). |

## Checks

For every recording here, `tests/test_scenario_files.py`:

- **replays it and compares with `<name>.expected.json`** (if present):

  ```json
  {
    "description": "What the recording shows.",
    "toggles": [false, true],
    "actions": ["START", "PERSON_FOUND", "GESTURE_STARTED", "FOLLOWING_STOPPED",
                "GESTURE_STARTED", "FOLLOWING_RESUMED", "PERSON_FOUND", "END"],
    "min_followed_fraction": 0.5
  }
  ```

  - `toggles` - following state after each stop / resume (`false` = stopped,
    `true` = resumed), in order.
  - `actions` - the exact sequence of actions in the log (see the main README
    for their meanings).
  - `min_followed_fraction` - minimum fraction of frames in which the person
    was followed and in view.
  - optional `stop_gesture` / `resume_gesture` / `draw_toggle_gesture` (e.g.
    `["PALM", "FIST"]`) - check under different gesture rules than the app's
    defaults.

- **checks the replay reproduces `<name>.live_actions.txt` exactly** (if
  present) - every action, in order, to the millisecond. A replay feeds the
  recorded MediaPipe output through the same logic with the same timestamps,
  so any difference means the logic (or its settings) has changed since the
  recording was made. If that change is intended, re-record the scenario.

Videos (`.mp4 .mov .avi .mkv .webm .m4v`) can be added too; MediaPipe is then
run on them, so their results can vary slightly between MediaPipe versions and
they are only compared with `<name>.expected.json`.

## Adding a scenario

1. Record it: `python follow_pose.py --record NAME` (stop with `q`).
2. Check `outputs/NAME/NAME.actions.txt` shows what you intended.
3. Copy `outputs/NAME/NAME.landmarks.jsonl` here, and
   `outputs/NAME/NAME.actions.txt` here as `NAME.live_actions.txt`.
4. Write `NAME.expected.json`: a description of what you did, and the
   `toggles` / `actions` from the action log.
5. Run `python -m pytest -m scenario`.

**Privacy:** videos contain people's faces, so `.gitignore` excludes video files
(`*.mp4`, `*.mov`, `*.avi`) and `outputs/` from git. Landmark files, logs and
expectations are tracked - they hold body-joint coordinates only, but check
that everyone recorded agrees.
