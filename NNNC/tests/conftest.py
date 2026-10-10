"""Put the strategies repo root (and the family dirs strategies import from) on sys.path,
the same way Framework/tests/ does."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "utils"), str(ROOT / "NNNC"), str(ROOT / "SimpleTrigger")):
    if p not in sys.path:
        sys.path.insert(0, p)
