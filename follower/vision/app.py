"""Main loop: pick an input, run detection + logic, draw, display, save outputs.

Inputs:
  1. the live camera, if one is available (``--record`` also saves the session);
  2. otherwise every recording under tests/scenarios/ (or ``--scenarios DIR``):
     videos, and ``.landmarks.jsonl`` MediaPipe recordings (replayed without video);
  3. ``--video PATH ...`` skips the camera and plays the given files/folders.

For every recording processed, ``outputs/<name>/`` receives the action log -
``<name>.actions.txt`` for videos (plus the extracted ``<name>.landmarks.jsonl``),
``<name>.replay.actions.txt`` for landmark recordings. Live sessions write files
only with ``--record``.
"""
import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2

from follower.logic import DEFAULT_SCENARIO_DIR, DEFAULT_SETTINGS, OUTPUTS_DIR, ActionLog, \
    PoseFollower, Settings, VIDEO_EXTENSIONS, find_scenarios, is_landmark_file, \
    scenario_stem

from .capture import LandmarkReplay, open_camera, open_replay, open_video
from .detector import Detection, MediaPipeDetector, NoDetector
from .drawing import render
from .outputs import SessionOutputs, describe_source

WINDOW = "Pose Follower"
QUIT_KEYS = (ord("q"), 27)  # q / ESC


@dataclass
class RunSummary:
    name: str
    frames: int = 0
    toggles: List[Tuple[float, bool]] = field(default_factory=list)  # (time, following after)
    quit: bool = False                     # user asked to quit everything
    log: Optional[ActionLog] = None        # the major actions detected
    files: List[Path] = field(default_factory=list)  # output files written

    @property
    def person_frames(self) -> int:
        """Frames where a followed person was in view."""
        return self.log.following_frames if self.log else 0


def display_available() -> bool:
    """False on a headless Linux box (Qt would abort the process on imshow)."""
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


def _detections(source, detector):
    """(image, t, Detection) per frame - from MediaPipe, or from a landmark recording."""
    if isinstance(source, LandmarkReplay):
        for canvas, t, rec in source.frames():
            yield canvas, t, Detection.from_record(rec)
    else:
        for img, t in source.frames():
            yield img, t, detector.process(img)


def run_source(source, display: bool = True, outputs: Optional[SessionOutputs] = None,
               settings: Settings = DEFAULT_SETTINGS, detector_factory=None,
               duration: Optional[float] = None) -> RunSummary:
    """Run the follower over one source until it ends, ``duration`` s pass, or the user quits.

    ``detector_factory`` exists so tests can substitute scripted detections.
    """
    summary = RunSummary(source.name)
    follower = PoseFollower(settings)
    log = ActionLog(settings)
    replay = isinstance(source, LandmarkReplay)
    started = time.monotonic()
    detector_factory = detector_factory or MediaPipeDetector
    with (NoDetector() if replay else detector_factory()) as detector:
        try:
            for index, (frame, now, det) in enumerate(_detections(source, detector)):
                if duration is not None and now > duration:
                    break
                h, w = frame.shape[:2]
                result = follower.update(det.pose_lm, det.hands, now, aspect=w / h)
                log.observe(result, now)
                if result.toggled:
                    # stop: let go of the old person; resume: lock onto the person
                    # who made the gesture (or whoever is in view if unknown)
                    detector.reset_pose()
                    detector.lock_on(result.resume_target if result.following else None)
                    summary.toggles.append((now, result.following))
                    print(f"  [{source.name} t={now:6.2f}s] "
                          f"{'FOLLOWING RESUMED' if result.following else 'FOLLOWING STOPPED'}")
                summary.frames += 1

                if outputs is not None:
                    outputs.write_input(index, now, frame, det)   # before drawing on it
                render(frame, det, result, settings, show_hands=replay)
                if outputs is not None:
                    outputs.write_annotated(frame, now)
                if display:
                    cv2.imshow(WINDOW, frame)
                    delay = 1
                    if not source.live:  # play files no faster than real time
                        delay = max(1, int((started + now - time.monotonic()) * 1000))
                    if cv2.waitKey(delay) & 0xFF in QUIT_KEYS:
                        summary.quit = True
                        break
        except KeyboardInterrupt:
            summary.quit = True
        finally:
            log.finish()
            summary.log = log
            if outputs is not None:
                summary.files = outputs.close(log, describe_source(source))
    return summary


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Follow the first person in view and draw their pose. Holding open palm + "
                    "fist for 3 s stops following; open palm + scissors resumes it (on whoever "
                    "shows it); fist + scissors shows/hides the pose drawing. Uses the live "
                    "camera if available, otherwise the recordings in tests/scenarios/.")
    p.add_argument("--camera", type=int, default=0, metavar="INDEX",
                   help="camera index to try first (default: 0)")
    p.add_argument("--video", nargs="+", type=Path, metavar="PATH",
                   help="skip the camera and play these videos / .landmarks.jsonl files / folders")
    p.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIO_DIR, metavar="DIR",
                   help="where to look for recordings when no camera is available "
                        f"(default: {DEFAULT_SCENARIO_DIR})")
    p.add_argument("--record", nargs="?", const="", metavar="NAME",
                   help="record a live camera session into OUTPUTS/NAME/: raw video, MediaPipe "
                        "landmarks (.landmarks.jsonl) and the actions log "
                        "(default NAME: rec_<date>_<time>)")
    p.add_argument("--duration", type=float, metavar="SECONDS",
                   help="stop after this many seconds (of footage)")
    p.add_argument("--outputs", type=Path, default=OUTPUTS_DIR, metavar="DIR",
                   help=f"where output files go (default: {OUTPUTS_DIR})")
    p.add_argument("--annotate", action="store_true",
                   help="also save <name>_annotated.mp4 (the pose diagram as displayed)")
    p.add_argument("--no-save", action="store_true",
                   help="don't write output files when playing recordings")
    p.add_argument("--no-display", action="store_true",
                   help="don't open a window (automatic when no display is available)")
    return p.parse_args(argv)


def _report(s: RunSummary):
    sm = s.log.summary()
    print(f"  {s.frames} frames, person followed in {100 * sm['followed_fraction']:.0f}%, "
          f"{sm['stops']} stop(s), {sm['resumes']} resume(s)")
    for f in s.files:
        print(f"  wrote {f}")


def _record(args, display) -> int:
    camera = open_camera(args.camera)
    if camera is None:
        print(f"--record needs a live camera, but none is available at index {args.camera}.",
              file=sys.stderr)
        return 1
    name = Path(args.record).name if args.record else time.strftime("rec_%Y%m%d_%H%M%S")
    folder = args.outputs / name
    if folder.exists() and any(folder.iterdir()):
        camera.release()
        print(f"{folder} already exists - choose another name.", file=sys.stderr)
        return 1
    how = "press q or ESC in the window" if display else "press Ctrl-C"
    print(f"Recording {camera.name} into {folder}/ - {how} to stop"
          + (f" (or wait {args.duration:g} s)." if args.duration else "."))
    outputs = SessionOutputs(folder, name, camera, raw_video=True, landmarks=True,
                             annotated=args.annotate)
    with camera:
        s = run_source(camera, display=display, outputs=outputs, duration=args.duration)
    if display:
        cv2.destroyAllWindows()
    _report(s)
    print(f"To use it as a test scenario copy {name}.landmarks.jsonl (no images, replays "
          f"exactly) and/or {name}.mp4 into {DEFAULT_SCENARIO_DIR}/")
    return 0


def _open_file(path: Path):
    return open_replay(path) if is_landmark_file(path) else open_video(path)


def main(argv=None) -> int:
    args = parse_args(argv)
    display = not args.no_display and display_available()
    if not display and not args.no_display:
        print("No display detected - running headless (events are printed below).")
    if args.record is not None:
        return _record(args, display)

    camera = open_camera(args.camera) if args.video is None else None
    if args.video is None and camera is None:
        print(f"No camera available at index {args.camera}; "
              f"looking for recorded scenarios in {args.scenarios}")
    if camera is not None:
        print(f"Using live {camera.name}. Press q or ESC to quit "
              f"(use --record to save the session).")
        with camera:
            s = run_source(camera, display=display, duration=args.duration)
        if display:
            cv2.destroyAllWindows()
        _report(s)
        return 0

    files = find_scenarios(args.video if args.video is not None else args.scenarios)
    if not files:
        where = ", ".join(map(str, args.video)) if args.video else args.scenarios
        print(f"No camera and no recordings found in {where}.\n"
              f"Put videos ({' '.join(sorted(VIDEO_EXTENSIONS))}) or .landmarks.jsonl files "
              f"there, or pass --video PATH.", file=sys.stderr)
        return 1
    print(f"Playing {len(files)} recording(s)" + (". Press q or ESC to quit." if display else ":"))

    failures = 0
    try:
        for path in files:
            try:
                source = _open_file(path)
            except IOError as e:
                print(f"  {e}", file=sys.stderr)
                failures += 1
                continue
            with source:
                print(f"- {path.name}")
                outputs = None
                if not args.no_save:
                    stem = scenario_stem(path)
                    replay = isinstance(source, LandmarkReplay)
                    outputs = SessionOutputs(args.outputs / stem,
                                             f"{stem}.replay" if replay else stem, source,
                                             landmarks=not replay, annotated=args.annotate)
                s = run_source(source, display=display, outputs=outputs, duration=args.duration)
            _report(s)
            if s.quit:
                break
    finally:
        if display:
            cv2.destroyAllWindows()
    return 1 if failures else 0

