"""Well-known project folders (pure pathlib)."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_DIR = PROJECT_ROOT / "tests" / "scenarios"   # recordings used as app input / tests
OUTPUTS_DIR = PROJECT_ROOT / "outputs"                # generated files (git-ignored)
