"""Reuse frozen portable3 scheduling/host modules; keep successor contracts first."""
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'portable_dispatcher_v3'
if str(HERE)not in sys.path:sys.path.insert(0,str(HERE))
if str(PARENT)not in sys.path:sys.path.insert(1,str(PARENT))
