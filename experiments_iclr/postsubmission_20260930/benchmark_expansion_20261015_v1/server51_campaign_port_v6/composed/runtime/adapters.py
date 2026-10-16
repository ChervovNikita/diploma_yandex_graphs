"""Genuine composed producer factory; the frozen native helper stays unchanged."""
from pathlib import Path
import sys
FOLDER = Path(__file__).resolve().parents[1]
if str(FOLDER) not in sys.path:
    sys.path.insert(0, str(FOLDER))
from runtime_bridge import create_adapter
