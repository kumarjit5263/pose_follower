"""Write every test case, with its inputs and outputs, to outputs/test_cases/.

    python -m tests.generate_cases            # -> outputs/test_cases/
    python -m tests.generate_cases --out DIR

For each case a folder holds:
  input     <case>.landmarks.jsonl (synthetic cases) or <case>.mp4 (real video)
  expected  <case>.expected.json   (description, expected toggles and actions)
  outputs   <case>.actions.txt     (major actions detected, written by the app)
            <case>_annotated.mp4   (the pose diagram, as the app displays it)
            <case>.landmarks.jsonl (MediaPipe output - video cases)
            <case>.replay.actions.txt (video cases: the MediaPipe output replayed
                                       without the video gives the same actions)
plus SUMMARY.md comparing expected vs detected for all cases.

Also included: the two-person lock-on case (real MediaPipe pose, scripted hands)
and every recording in tests/scenarios/ that has a <name>.expected.json (with
its live action log, if any, next to the replayed one).

Test-only code: needs cv2 + mediapipe, and scikit-image for the astronaut photo.
"""
import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from follower.logic import OUTPUTS_DIR, SCENARIO_DIR, ActionKind as K, find_scenarios, \
    scenario_stem
from tests.cases import CASES, write_landmarks

ASTRONAUT = dict(
    name="astronaut_walkthrough",
    description="Real MediaPipe on a real photo (NASA astronaut portrait, public domain, "
                "bundled with scikit-image): the person drifts side to side for 3 s, "
                "leaves the frame for 1 s, and comes back for 2 s. No hands are shown.",
    expected_toggles=[],
    expected_actions=[K.START, K.PERSON_FOUND, K.PERSON_LOST, K.PERSON_FOUND, K.END],
)


def make_astronaut_video(path, seconds_in=3.0, seconds_out=1.0, seconds_back=2.0, fps=30,
                         size=(640, 480)):
    """Astronaut photo drifting side to side, absent for ``seconds_out`` in the middle."""
    import cv2
    from skimage import data

    w, h = size
    person = cv2.resize(cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR), (512, 512))[32:32 + h]
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n_in, n_out = round(seconds_in * fps), round(seconds_out * fps)
    for i in range(n_in + n_out + round(seconds_back * fps)):
        canvas = np.full((h, w, 3), 60, np.uint8)
        if not n_in <= i < n_in + n_out:
            x0 = 64 + int(40 * np.sin(i / 15))
            canvas[:, x0:x0 + 512] = person
        out.write(canvas)
    out.release()
    return Path(path)


TWO_PEOPLE = dict(
    name="resume_locks_onto_gesturer",
    description="Real MediaPipe pose on two people (astronaut photo twice: large on the "
                "left, smaller on the right); hands are scripted. Left person is followed "
                "and stops with open palm + fist; the RIGHT person then resumes with open palm "
                "+ scissors - following must "
                "switch to the right person (before the fix: the more prominent left one).",
    expected_toggles=[False, True],
    expected_actions=[K.START, K.PERSON_FOUND, K.GESTURE_STARTED, K.STOPPED,
                      K.GESTURE_STARTED, K.RESUMED, K.PERSON_FOUND, K.END],
)
TWO_PEOPLE_SIZE = (1280, 720)


def make_two_people_video(path, seconds=11.0, fps=30):
    """Static frame: a large person on the left and a smaller one on the right."""
    import cv2
    from skimage import data

    w, h = TWO_PEOPLE_SIZE
    ast = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
    frame = np.full((h, w, 3), 60, np.uint8)
    frame[80:720, 20:660] = cv2.resize(ast, (640, 640))        # left: large, most prominent
    frame[300:720, 800:1220] = cv2.resize(ast, (420, 420))     # right: smaller
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for _ in range(round(seconds * fps)):
        out.write(frame)
    out.release()
    return Path(path)


def two_people_hand_script(fps=30):
    """Per-frame scripted hands: left person's open palm + fist 1-4.5 s (stop), nothing,
    then the right person's open palm + scissors 5.5-9 s (resume), then nothing."""
    from follower.logic import Gesture
    from tests.landmarks import make_hand
    aspect = TWO_PEOPLE_SIZE[0] / TWO_PEOPLE_SIZE[1]
    left = [make_hand(Gesture.FIST, 0.15, 0.75, size=0.12, aspect=aspect),       # stop
            make_hand(Gesture.PALM, 0.38, 0.75, size=0.12, aspect=aspect)]
    right = [make_hand(Gesture.SCISSORS, 0.70, 0.72, size=0.08, aspect=aspect),   # resume
             make_hand(Gesture.PALM, 0.84, 0.72, size=0.08, aspect=aspect)]
    n = lambda s: round(s * fps)  # noqa: E731
    return [[]] * n(1) + [left] * n(3.5) + [[]] * n(1) + [right] * n(3.5) + [[]] * n(2)


class ScriptedHandsDetector:
    """Real MediaPipe pose (incl. lock-on), but hands come from a script - there is
    no real hand footage to test with."""

    def __init__(self, script):
        from follower.vision.detector import MediaPipeDetector
        self._det = MediaPipeDetector(recover_hands=False)
        frames = iter(script)

        def hands_process(rgb):
            from mediapipe.framework.formats import landmark_pb2
            hands = next(frames, [])
            return SimpleNamespace(multi_hand_landmarks=[
                landmark_pb2.NormalizedLandmarkList(landmark=[
                    landmark_pb2.NormalizedLandmark(x=p.x, y=p.y, z=p.z) for p in h])
                for h in hands] or None)
        self._det._hands.process = hands_process
        self.followed_x = []      # shoulder-midpoint x of the pose on each frame

    def process(self, frame):
        d = self._det.process(frame)
        lm = d.pose_lm
        self.followed_x.append(None if lm is None else (lm[11].x + lm[12].x) / 2)
        return d

    def reset_pose(self):
        self._det.reset_pose()

    def lock_on(self, target):
        self._det.lock_on(target)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._det.close()


def run_two_people(folder, name, annotate=True):
    """Returns (RunSummary, ScriptedHandsDetector)."""
    from follower.vision.app import run_source
    from follower.vision.capture import open_video
    from follower.vision.outputs import SessionOutputs

    video = make_two_people_video(Path(folder) / f"{name}.mp4")
    det = ScriptedHandsDetector(two_people_hand_script())
    with open_video(video) as src:
        outs = SessionOutputs(folder, name, src, landmarks=True, annotated=annotate)
        s = run_source(src, display=False, outputs=outs, detector_factory=lambda: det)
    return s, det


def _run(path, folder, name, annotate=True, settings=None):
    from follower.logic import DEFAULT_SETTINGS
    from follower.vision.app import run_source
    from follower.vision.capture import open_replay, open_video
    from follower.vision.outputs import SessionOutputs

    settings = settings or DEFAULT_SETTINGS
    replay = path.name.endswith(".landmarks.jsonl")
    with (open_replay(path) if replay else open_video(path)) as src:
        outs = SessionOutputs(folder, name, src, landmarks=not replay, annotated=annotate)
        return run_source(src, display=False, outputs=outs, settings=settings)


def _row(name, description, expected_toggles, expected_actions, s):
    got_toggles = [f for _, f in s.toggles]
    ok = got_toggles == expected_toggles and s.log.kinds() == expected_actions
    fmt = lambda ts: ", ".join("resume" if f else "stop" for f in ts) or "none"  # noqa: E731
    when = ", ".join(f"{'resume' if f else 'stop'} @ {t:.2f}s" for t, f in s.toggles) or "none"
    return ok, (f"| {'PASS' if ok else '**FAIL**'} | `{name}` | {description} | "
                f"{fmt(expected_toggles)} | {when} |")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=OUTPUTS_DIR / "test_cases")
    args = p.parse_args(argv)
    out = args.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    rows, failures = [], 0
    for case in CASES:
        folder = out / case.name
        folder.mkdir()
        src = folder / f"{case.name}.landmarks.jsonl"
        write_landmarks(case, src)
        (folder / f"{case.name}.expected.json").write_text(json.dumps({
            "description": case.description,
            "toggles": case.expected_toggles,
            "actions": [k.value for k in case.expected_actions]}, indent=2) + "\n")
        print(f"- {case.name}")
        s = _run(src, folder, case.name)
        ok, row = _row(case.name, case.description, case.expected_toggles,
                       case.expected_actions, s)
        failures += not ok
        rows.append(row)

    if importlib.util.find_spec("skimage") is None:
        print("scikit-image not installed - skipping the astronaut case "
              "(pip install -r requirements-dev.txt)", file=sys.stderr)
        failures += 1
    else:
        a = ASTRONAUT
        folder = out / a["name"]
        folder.mkdir()
        video = make_astronaut_video(folder / f"{a['name']}.mp4")
        (folder / f"{a['name']}.expected.json").write_text(json.dumps({
            "description": a["description"], "toggles": a["expected_toggles"],
            "actions": [k.value for k in a["expected_actions"]]}, indent=2) + "\n")
        print(f"- {a['name']} (real MediaPipe)")
        s = _run(video, folder, a["name"])
        ok, row = _row(a["name"], a["description"], a["expected_toggles"],
                       a["expected_actions"], s)
        # replay the MediaPipe output that was just written: must give the same actions
        r = _run(folder / f"{a['name']}.landmarks.jsonl", folder, f"{a['name']}.replay",
                 annotate=False)
        same = [(x.t, x.kind) for x in r.log.actions] == [(x.t, x.kind) for x in s.log.actions]
        ok = ok and same
        row = row.replace("| PASS |", "| PASS |" if ok else "| **FAIL** |", 1)
        failures += not ok
        rows.append(row + ("" if same else " (replay differs!)"))

        # two people: resuming must lock onto the person who made the gesture
        t = TWO_PEOPLE
        folder = out / t["name"]
        folder.mkdir()
        (folder / f"{t['name']}.expected.json").write_text(json.dumps({
            "description": t["description"], "toggles": t["expected_toggles"],
            "actions": [k.value for k in t["expected_actions"]]}, indent=2) + "\n")
        print(f"- {t['name']} (real MediaPipe pose, scripted hands)")
        s, det = run_two_people(folder, t["name"])
        ok, row = _row(t["name"], t["description"], t["expected_toggles"],
                       t["expected_actions"], s)
        resume_t = s.toggles[-1][0] if s.toggles else None
        fps = 30
        after = [x for i, x in enumerate(det.followed_x)
                 if resume_t is not None and i / fps > resume_t and x is not None]
        before = [x for x in det.followed_x[:fps] if x is not None]
        locked = bool(after) and all(x > 0.6 for x in after) and all(x < 0.5 for x in before)
        ok = ok and locked
        row = row.replace("| PASS |", "| PASS |" if ok else "| **FAIL** |", 1)
        failures += not ok
        if after and before:
            row = row[:-2] + (f"; followed the left person (x~{np.mean(before):.2f}) before, "
                              f"the right person (x~{np.mean(after):.2f}) after resuming |")
        rows.append(row)

    # recordings in tests/scenarios/ that come with expectations
    from follower.logic import Settings
    overridable = ("stop_gesture", "resume_gesture", "draw_toggle_gesture")
    for path in find_scenarios(SCENARIO_DIR):
        stem = scenario_stem(path)
        exp_file = path.with_name(f"{stem}.expected.json")
        if not exp_file.exists():
            continue
        expected = json.loads(exp_file.read_text())
        folder = out / "scenarios" / stem
        folder.mkdir(parents=True)
        shutil.copy(exp_file, folder / exp_file.name)
        live_log = path.with_name(f"{stem}.live_actions.txt")
        if live_log.exists():                      # for side-by-side comparison
            shutil.copy(live_log, folder / live_log.name)
        print(f"- scenarios/{path.name}")
        overrides = {k: tuple(expected[k]) for k in overridable if k in expected}
        name = f"{stem}.replay" if path.name.endswith(".landmarks.jsonl") else stem
        s = _run(path, folder, name, settings=Settings(**overrides) if overrides else None)
        exp_actions = [K(a) for a in expected.get("actions", [])] or s.log.kinds()
        ok, row = _row(f"scenarios/{path.name}", expected.get("description", ""),
                       expected.get("toggles", [f for _, f in s.toggles]), exp_actions, s)
        if "min_followed_fraction" in expected:
            ok = ok and s.person_frames / s.frames >= expected["min_followed_fraction"]
            row = row.replace("| PASS |", "| PASS |" if ok else "| **FAIL** |", 1)
        failures += not ok
        rows.append(row)

    (out / "SUMMARY.md").write_text("\n".join([
        "# pose_follower test cases",
        "",
        "Generated by `python -m tests.generate_cases`. Each folder holds the input, "
        "`*.expected.json`, and the files the app produced (`*.actions.txt` = major "
        "actions detected, `*_annotated.mp4` = pose diagram as displayed, "
        "`*.landmarks.jsonl` = MediaPipe output for video inputs).",
        "",
        "| Result | Case | Scenario | Expected toggles | Detected |",
        "|---|---|---|---|---|",
        *rows,
        "",
        f"**{len(rows) - failures} / {len(rows)} passed.**",
        ""]), encoding="utf-8")
    print(f"\n{len(rows) - failures}/{len(rows)} cases passed - see {out / 'SUMMARY.md'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
