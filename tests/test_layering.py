"""Guards the code split: follower.logic must stay free of cv2 / mediapipe,
and the app package must never depend on test code."""
import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_IN_LOGIC = {"cv2", "mediapipe", "follower.vision"}


def _imports(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module


def _bad(names, forbidden):
    return [n for n in names if any(n == f or n.startswith(f + ".") for f in forbidden)]


def test_logic_sources_do_not_import_cv2_or_mediapipe():
    offenders = {str(p.relative_to(ROOT)): _bad(_imports(p), FORBIDDEN_IN_LOGIC)
                 for p in (ROOT / "follower" / "logic").rglob("*.py")}
    assert {k: v for k, v in offenders.items() if v} == {}


def test_importing_logic_does_not_load_cv2_or_mediapipe():
    code = ("import sys, follower.logic; "
            "bad = [m for m in ('cv2', 'mediapipe') if m in sys.modules]; "
            "print(bad); sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_app_code_does_not_import_tests():
    offenders = {str(p.relative_to(ROOT)): _bad(_imports(p), {"tests", "pytest"})
                 for p in list((ROOT / "follower").rglob("*.py")) + [ROOT / "follow_pose.py"]}
    assert {k: v for k, v in offenders.items() if v} == {}
