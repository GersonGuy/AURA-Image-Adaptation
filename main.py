import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "tta_lib" / "test-time-adaptation"

sys.path.insert(0, str(REPO))

print("python exe:", sys.executable)
print("repo path:", REPO)

import classification
print("classification OK")