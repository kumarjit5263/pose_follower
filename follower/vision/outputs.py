"""Files written for a session into ``outputs/<name>/``.

- ``<name>.mp4``               raw footage (live recordings only)
- ``<name>.landmarks.jsonl``   MediaPipe output per frame (camera / video input)
- ``<name>.actions.txt``       timeline of the major actions + summary
- ``<name>_annotated.mp4``     what the window showed (optional, ``--annotate``)
"""
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2

from follower.logic import ActionLog, LandmarkFileWriter, format_report

from .detector import MEDIAPIPE_VERSION, Detection

FOURCC = cv2.VideoWriter_fourcc(*"mp4v")


class VideoOut:
    """Writes frames to an .mp4 whose frame rate matches the frames' real timestamps.

    A live camera rarely delivers its nominal fps (and processing slows it
    further), so the video is written at the nominal rate and, if the measured
    rate differs by more than 5 %, re-timed on close. Otherwise a 3 s gesture
    in the recording would not last 3 s when the video is played back.
    """

    def __init__(self, path, fps: float):
        self.path = Path(path)
        self.fps = fps
        self._tmp = self.path.with_name(self.path.stem + ".partial" + self.path.suffix)
        self._writer = None
        self._n, self._t0, self._t1 = 0, None, None

    def write(self, img, t: float):
        if self._writer is None:
            h, w = img.shape[:2]
            self._writer = cv2.VideoWriter(str(self._tmp), FOURCC, self.fps, (w, h))
            if not self._writer.isOpened():
                raise IOError(f"Cannot write video {self._tmp}")
        self._writer.write(img)
        self._n += 1
        self._t0 = t if self._t0 is None else self._t0
        self._t1 = t

    def close(self) -> Optional[Path]:
        if self._writer is None:
            return None
        self._writer.release()
        self._writer = None
        measured = (self._n - 1) / (self._t1 - self._t0) if self._n > 1 and self._t1 > self._t0 \
            else self.fps
        if abs(measured / self.fps - 1) > 0.05:
            self._retime(measured)
        else:
            self._tmp.replace(self.path)
        return self.path

    def _retime(self, fps):
        src = cv2.VideoCapture(str(self._tmp))
        dst = None
        while True:
            ok, img = src.read()
            if not ok:
                break
            if dst is None:
                h, w = img.shape[:2]
                dst = cv2.VideoWriter(str(self.path), FOURCC, fps, (w, h))
            dst.write(img)
        src.release()
        if dst is not None:
            dst.release()
        self._tmp.unlink(missing_ok=True)
        self.fps = fps


def describe_source(source) -> Dict[str, object]:
    kind = {"camera": "live camera", "video": "video file",
            "landmarks": "landmark recording (replayed, no video)"}[source.kind]
    return {"source": source.name, "input": kind,
            "frame size": f"{source.width}x{source.height}",
            "created": time.strftime("%Y-%m-%d %H:%M:%S")}


class SessionOutputs:
    def __init__(self, folder, name: str, source, *, raw_video=False, landmarks=False,
                 annotated=False):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.raw = VideoOut(self.folder / f"{name}.mp4", source.fps) if raw_video else None
        self.annotated = VideoOut(self.folder / f"{name}_annotated.mp4", source.fps) \
            if annotated else None
        self.landmarks = None
        if landmarks:
            self.landmarks = LandmarkFileWriter(
                self.folder / f"{name}.landmarks.jsonl", source=source.name, kind=source.kind,
                width=source.width, height=source.height, fps=source.fps,
                mirrored=source.mirror, mediapipe=MEDIAPIPE_VERSION,
                created=time.strftime("%Y-%m-%dT%H:%M:%S"))
        self.actions_path = self.folder / f"{name}.actions.txt"

    def write_input(self, index: int, t: float, frame, det: Detection):
        """Call before drawing on ``frame``."""
        if self.raw is not None:
            self.raw.write(frame, t)
        if self.landmarks is not None:
            self.landmarks.write(index, t, det.pose_lm, det.hands)

    def write_annotated(self, frame, t: float):
        if self.annotated is not None:
            self.annotated.write(frame, t)

    def close(self, log: ActionLog, source_info: Dict[str, object]) -> List[Path]:
        files = []
        for video in (self.raw, self.annotated):
            if video is not None and (p := video.close()) is not None:
                files.append(p)
        if self.landmarks is not None:
            self.landmarks.close()
            files.append(self.landmarks.path)
        self.actions_path.write_text(format_report(log, source_info), encoding="utf-8")
        files.append(self.actions_path)
        return files
