"""Read / write MediaPipe landmark recordings (``*.landmarks.jsonl``).

The file stores what MediaPipe detected in every frame, so a session can be
replayed through the follow / gesture logic later without the video (and
without re-running MediaPipe). It contains no images.

Format (JSON Lines):
  line 1  header: {"format": "pose_follower.landmarks", "version": 1,
                   "source", "kind", "width", "height", "fps", "mirrored", ...}
  line 2+ one frame each: {"i": index, "t": seconds,
                   "pose":  [[x, y, z, visibility] x 33] or null,
                   "hands": [[[x, y, z] x 21], ...]}
Coordinates are MediaPipe's normalised image coordinates. ``fps`` is the
frames' measured rate; if a camera reported a different one, it's kept as
``nominal_fps``.

Numbers are written with 9 significant digits and read back as float32, which
reproduces MediaPipe's float32 values exactly - replaying a recording makes
exactly the same decisions as the live session did.
"""
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional

FORMAT = "pose_follower.landmarks"
VERSION = 1
SUFFIX = ".landmarks.jsonl"

_F32 = struct.Struct("f")


def _f32(v: float) -> float:
    return _F32.unpack(_F32.pack(v))[0]


def _num(v: float) -> float:
    return float(f"{v:.9g}")


@dataclass
class Landmark:
    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0


@dataclass
class FrameRecord:
    index: int
    t: float
    pose: Optional[List[Landmark]] = None
    hands: List[List[Landmark]] = field(default_factory=list)


class LandmarkFileWriter:
    """Streams frames to a ``.landmarks.jsonl`` file.

    ``pose`` / ``hands`` are duck-typed landmark sequences (MediaPipe protos'
    ``.landmark`` lists or ``Landmark`` objects).
    """

    def __init__(self, path, *, source: str, kind: str, width: int, height: int,
                 fps: float, mirrored: bool = False, **extra):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("w", encoding="utf-8")
        self.header = {"format": FORMAT, "version": VERSION, "source": source, "kind": kind,
                       "width": int(width), "height": int(height), "fps": float(fps),
                       "mirrored": bool(mirrored), **extra}
        self._f.write(json.dumps(self.header) + "\n")
        self.frames = 0
        self._t0 = self._t1 = None

    def write(self, index: int, t: float, pose=None, hands=()):
        rec = {
            "i": index,
            "t": float(t),  # full precision: replay uses the exact same clock
            "pose": None if pose is None else
            [[_num(p.x), _num(p.y), _num(p.z), _num(p.visibility)] for p in pose],
            "hands": [[[_num(p.x), _num(p.y), _num(p.z)] for p in hand] for hand in hands],
        }
        self._f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        self.frames += 1
        self._t0 = t if self._t0 is None else self._t0
        self._t1 = t

    def close(self):
        """Finish the file. If the frames' real rate differs from the header's
        ``fps`` by > 5 % (cameras often report a wrong or nominal rate), the
        header is rewritten with the measured rate (the nominal one is kept as
        ``nominal_fps``)."""
        if self._f.closed:
            return
        self._f.close()
        if self.frames > 1 and self._t1 > self._t0:
            measured = (self.frames - 1) / (self._t1 - self._t0)
            nominal = self.header["fps"]
            if nominal <= 0 or abs(measured / nominal - 1) > 0.05:
                self.header = {**self.header, "fps": round(measured, 3),
                               "nominal_fps": self.header["fps"]}
                lines = self.path.read_text(encoding="utf-8").splitlines(keepends=True)
                lines[0] = json.dumps(self.header) + "\n"
                self.path.write_text("".join(lines), encoding="utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class LandmarkFile:
    """A recording opened for reading: ``header`` plus a ``frames()`` iterator."""

    def __init__(self, path):
        self.path = Path(path)
        with self.path.open(encoding="utf-8") as f:
            first = f.readline()
        try:
            header = json.loads(first)
        except json.JSONDecodeError:
            header = None
        if not isinstance(header, dict) or header.get("format") != FORMAT:
            raise ValueError(f"{self.path} is not a pose_follower landmarks file")
        if header.get("version") != VERSION:
            raise ValueError(f"{self.path}: unsupported version {header.get('version')}")
        self.header = header

    @property
    def width(self) -> int:
        return int(self.header.get("width") or 640)

    @property
    def height(self) -> int:
        return int(self.header.get("height") or 480)

    @property
    def fps(self) -> float:
        return float(self.header.get("fps") or 30.0)

    def frames(self) -> Iterator[FrameRecord]:
        with self.path.open(encoding="utf-8") as f:
            f.readline()  # header
            for n, line in enumerate(f, start=2):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    pose = rec.get("pose")
                    yield FrameRecord(
                        index=int(rec["i"]),
                        t=float(rec["t"]),
                        pose=None if pose is None else
                        [Landmark(_f32(x), _f32(y), _f32(z), _f32(v)) for x, y, z, v in pose],
                        hands=[[Landmark(_f32(x), _f32(y), _f32(z)) for x, y, z in hand]
                               for hand in rec.get("hands", [])],
                    )
                except (KeyError, TypeError, ValueError) as e:
                    raise ValueError(f"{self.path}:{n}: bad frame record ({e})") from e


def is_landmark_file(path) -> bool:
    return Path(path).name.lower().endswith(SUFFIX)
