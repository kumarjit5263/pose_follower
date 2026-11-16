#!/usr/bin/env bash
# (Re)create the Python 3.12 virtual environment for pose_follower.
#
# Safe to run any time: if .venv is missing, broken, or on the wrong Python
# version it is rebuilt; otherwise dependencies are just (re)installed.
#
# Usage:
#   ./setup_env.sh              # create/repair .venv, install app + test deps
#   ./setup_env.sh --recreate   # delete .venv and build it from scratch
#   ./setup_env.sh --no-lock    # ignore requirements.lock, resolve latest allowed
#   ./setup_env.sh --runtime-only   # skip test dependencies (pytest)
#   PYTHON=/path/to/python3.12 ./setup_env.sh   # use a specific interpreter
#
# Afterwards:  source .venv/bin/activate && python follow_pose.py
set -euo pipefail

REQUIRED_PY="3.12"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
LOCK_FILE="$PROJECT_DIR/requirements.lock"

recreate=0; use_lock=1; req_file="requirements-dev.txt"
for arg in "$@"; do
  case "$arg" in
    --recreate)     recreate=1 ;;
    --no-lock)      use_lock=0 ;;
    --runtime-only) req_file="requirements.txt" ;;
    -h|--help)      awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "Unknown option: $arg (see --help)" >&2; exit 2 ;;
  esac
done

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mWARNING:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

py_version() { "$1" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null; }

# ---- 1. find a Python 3.12 interpreter --------------------------------------
find_python() {
  if [[ -n "${PYTHON:-}" ]]; then  # explicit choice: use it or fail, never silently ignore
    [[ "$(py_version "$PYTHON")" == "$REQUIRED_PY" ]] && { command -v "$PYTHON"; return 0; }
    echo "PYTHON=$PYTHON is not a Python $REQUIRED_PY interpreter." >&2; return 1
  fi
  for c in python3.12 python3 python; do
    if command -v "$c" >/dev/null 2>&1 && [[ "$(py_version "$c")" == "$REQUIRED_PY" ]]; then
      command -v "$c"; return 0
    fi
  done
  # Fall back to uv, which can download a standalone Python if none is installed.
  # (--system + running outside the project stops uv from returning .venv's own
  # interpreter, which --recreate would delete.)
  if command -v uv >/dev/null 2>&1; then
    local p
    uv_find() { (cd / && VIRTUAL_ENV= uv python find --system "$REQUIRED_PY" 2>/dev/null) || true; }
    p="$(uv_find)"
    if [[ -z "$p" ]]; then
      echo "Python $REQUIRED_PY not found; installing it with uv..." >&2
      uv python install "$REQUIRED_PY" >&2 && p="$(uv_find)"
    fi
    if [[ -n "$p" && "$p" != "$VENV_DIR"/* ]]; then echo "$p"; return 0; fi
  fi
  return 1
}

PY="$(find_python)" || die "Python $REQUIRED_PY not found.
  Install it, e.g.:
    Ubuntu/Debian: sudo apt install python3.12 python3.12-venv
    macOS:         brew install python@3.12
    any OS:        https://docs.astral.sh/uv/  then re-run this script
  or point PYTHON at an interpreter:  PYTHON=/path/to/python3.12 $0"
log "Using $PY (Python $(py_version "$PY"))"

# ---- 2. create (or repair) the venv -----------------------------------------
if [[ -d "$VENV_DIR" && $recreate -eq 0 ]]; then
  if [[ "$(py_version "$VENV_DIR/bin/python")" != "$REQUIRED_PY" ]] \
     || ! "$VENV_DIR/bin/python" -m pip --version >/dev/null 2>&1; then
    warn "Existing .venv is broken or not Python $REQUIRED_PY - rebuilding it."
    recreate=1
  fi
fi
if [[ $recreate -eq 1 && -d "$VENV_DIR" ]]; then
  log "Removing $VENV_DIR"
  rm -rf "$VENV_DIR"
fi
if [[ ! -d "$VENV_DIR" ]]; then
  log "Creating virtual environment in $VENV_DIR"
  "$PY" -m venv "$VENV_DIR" || die "venv creation failed.
  On Debian/Ubuntu you may need:  sudo apt install python3.12-venv"
fi
VPY="$VENV_DIR/bin/python"

# ---- 3. install dependencies ------------------------------------------------
log "Upgrading pip"
"$VPY" -m pip install --quiet --upgrade pip

cd "$PROJECT_DIR"
installed=0
if [[ $use_lock -eq 1 && -f "$LOCK_FILE" ]]; then
  log "Installing $req_file pinned to known-good versions in requirements.lock"
  if "$VPY" -m pip install -r "$req_file" -c "$LOCK_FILE"; then
    installed=1
  else
    warn "Pinned install failed (e.g. no wheels for this OS/CPU); retrying unpinned."
  fi
fi
if [[ $installed -eq 0 ]]; then
  log "Installing $req_file (latest versions allowed by the requirement ranges)"
  "$VPY" -m pip install -r "$req_file"
fi

# ---- 4. verify ----------------------------------------------------------------
log "Verifying imports"
if ! "$VPY" - <<'EOF'
import cv2, mediapipe as mp, numpy
assert hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"), \
    "this mediapipe build lacks the legacy mp.solutions API (need < 0.10.30)"
import follower.logic, follower.vision  # the app's own packages import cleanly
print(f"  python {__import__('sys').version.split()[0]} | mediapipe {mp.__version__} "
      f"| opencv {cv2.__version__} | numpy {numpy.__version__}")
EOF
then
  die "Import check failed. If the error mentions libGL.so.1 / libgthread, install
  the system OpenCV runtime libs, e.g.:  sudo apt install libgl1 libglib2.0-0"
fi

log "Done. Activate with:   source .venv/bin/activate"
echo "    run the app:        python follow_pose.py        (live camera, else the recordings in tests/scenarios)"
[[ "$req_file" == "requirements-dev.txt" ]] && echo "    run the tests:      python -m pytest"
exit 0
