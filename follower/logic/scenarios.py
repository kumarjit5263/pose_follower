"""Locate recorded test scenarios: videos and landmark recordings (pure pathlib)."""
from pathlib import Path
from typing import Iterable, List, Union

from .landmark_file import SUFFIX as LANDMARK_SUFFIX, is_landmark_file
from .paths import SCENARIO_DIR

VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"})
DEFAULT_SCENARIO_DIR = SCENARIO_DIR

PathLike = Union[str, Path]


def is_video_file(path) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def is_scenario_file(path) -> bool:
    return is_video_file(path) or is_landmark_file(path)


def scenario_stem(path) -> str:
    """'walk.mp4' -> 'walk', 'walk.landmarks.jsonl' -> 'walk'."""
    name = Path(path).name
    if is_landmark_file(name):
        return name[: -len(LANDMARK_SUFFIX)]
    return Path(name).stem


def find_scenarios(paths: Union[PathLike, Iterable[PathLike]] = DEFAULT_SCENARIO_DIR) -> List[Path]:
    """Videos and ``*.landmarks.jsonl`` files in the given file(s)/folder(s).

    Folders are searched recursively; missing paths are ignored; sorted.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    found = set()
    for p in map(Path, paths):
        if p.is_file():
            if is_scenario_file(p):
                found.add(p)
        elif p.is_dir():
            found.update(f for f in p.rglob("*") if f.is_file() and is_scenario_file(f))
    return sorted(found)
