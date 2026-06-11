"""Root conftest — makes `neurolink` importable when pytest runs from the repo root."""
import sys
from pathlib import Path

# Add backend/src to the import path so `import neurolink` works without
# having to `cd backend` or install the package first.
sys.path.insert(0, str(Path(__file__).parent / "backend" / "src"))
