"""Access to the third-party source trees kept outside the package.

third_party/minigpt4 is upstream MiniGPT-4, unmodified. It uses absolute
imports (`from minigpt4.common...`), so it cannot be imported as a subpackage
of mllmsent; it is put on sys.path on demand instead. Nothing in mllmsent
imports it — the captions it produced are already in data/minigpt4-classify —
but it is kept so those captions can be regenerated.
"""

from __future__ import annotations

import sys
from pathlib import Path

THIRD_PARTY_ROOT = Path(__file__).resolve().parents[2] / "third_party"


def enable_minigpt4() -> Path:
    """Puts third_party/ on sys.path so `import minigpt4` resolves."""
    root = THIRD_PARTY_ROOT
    if not (root / "minigpt4").is_dir():
        raise SystemExit(f"MiniGPT-4 source not found at {root / 'minigpt4'}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root
