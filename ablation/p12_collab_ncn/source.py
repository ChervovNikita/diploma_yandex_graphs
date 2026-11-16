"""Import the unchanged official modules while containing their legacy import."""
import sys
from external_sources.ncn import utils as official_utils

# Upstream model.py imports its sibling as the top-level name "utils".
# Temporarily supply that exact sibling, then restore the process namespace.
_previous_utils = sys.modules.get("utils")
sys.modules["utils"] = official_utils
try:
    from external_sources.ncn.model import CNLinkPredictor, GCN
finally:
    if _previous_utils is None:
        del sys.modules["utils"]
    else:
        sys.modules["utils"] = _previous_utils
