"""Input sources: a live camera, a video file, or a landmark recording.

``FrameSource.frames()`` yields ``(bgr_image, timestamp_seconds)``:
- camera: seconds since the first frame (wall clock), image mirrored (selfie view);
- video file: the frame's position in the video (frame_index / fps), so the
  gesture timing behaves the same however fast or slow processing runs.

``LandmarkReplay.frames()`` yields ``(blank_canvas, timestamp, FrameRecord)``
from a ``.landmarks.jsonl`` recording - no video or MediaPipe needed.
"""
import math
import time
from pathlib import Path
from typing import Iterator, Tuple

import cv2
import numpy as np

from follower.logic import FrameRecord, LandmarkFile

DEFAULT_FPS = 30.0
CANVAS_COLOR = (40, 40, 40)


class FrameSource:
    def __init__(self, cap: "cv2.VideoCapture", name: str, live: bool, mirror: bool):
        self._cap = cap
        self.name = name
        self.live = live
        self.mirror = mirror
        self.kind = "camera" if live else "video"
        self._pending = None  # frame read while probing the camera
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and math.isfinite(fps) and 0 < fps <= 1000 else DEFAULT_FPS
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    def frames(self) -> Iterator[Tuple[np.ndarray, float]]:
        index, t0 = 0, None
        while True:
            if self._pending is not None:
                ok, img, self._pending = True, self._pending, None
            else:
                ok, img = self._cap.read()
            if not ok or img is None:
                return  # end of file, or camera unplugged
            if self.mirror:
                img = cv2.flip(img, 1)
            self.height, self.width = img.shape[:2]
            if self.live:
                now = time.monotonic()
                t0 = now if t0 is None else t0
                t = now - t0
            else:
                t = index / self.fps
            yield img, t
            index += 1

    def release(self):
        self._cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


class LandmarkReplay:
    """Plays back a ``.landmarks.jsonl`` recording on a blank canvas."""

    live = False
    mirror = False
    kind = "landmarks"

    def __init__(self, path):
        self.file = LandmarkFile(path)   # raises ValueError if it isn't one
        self.name = Path(path).name
        self.fps = self.file.fps
        self.width, self.height = self.file.width, self.file.height

    def frames(self) -> Iterator[Tuple[np.ndarray, float, FrameRecord]]:
        for rec in self.file.frames():
            canvas = np.empty((self.height, self.width, 3), np.uint8)
            canvas[:] = CANVAS_COLOR
            yield canvas, rec.t, rec

    def release(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass


def open_camera(index: int = 0):
    """Open camera ``index`` if it exists *and* delivers a frame, else None."""
    old_level = cv2.setLogLevel(0)  # silence "can't open camera" noise while probing
    try:
        cap = cv2.VideoCapture(index)
        ok, img = cap.read() if cap.isOpened() else (False, None)
    finally:
        cv2.setLogLevel(old_level)
    if not ok or img is None:
        cap.release()
        return None
    src = FrameSource(cap, f"camera {index}", live=True, mirror=True)
    src._pending = img
    src.height, src.width = img.shape[:2]
    return src


def open_video(path) -> FrameSource:
    """Open a video file. Raises IOError if OpenCV can't read it."""
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise IOError(f"Cannot open video {path}")
    return FrameSource(cap, path.name, live=False, mirror=False)


def open_replay(path) -> LandmarkReplay:
    """Open a landmark recording. Raises IOError if it can't be read."""
    try:
        return LandmarkReplay(path)
    except (OSError, ValueError) as e:
        raise IOError(f"Cannot open landmark recording {path}: {e}") from e
