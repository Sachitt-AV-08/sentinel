"""Test configuration.

Adds the `src` layout to `sys.path` and, when orvima is available locally,
its `src` too - so the contract tests in `test_orvima_contract.py` exercise the
real tool surface instead of a copy of it.

orvima is located, in order, from:

1. `$ORVIMA_SRC`, if set
2. `../orvima/src` - the usual sibling-clone layout
3. `$HOME/orvima/src` - the Windows default

When none of those exist the contract tests skip rather than fail, so Sentinel's
suite still runs on its own. That is the right trade: a checkout without orvima
is not a broken Sentinel, it is a Sentinel with nothing to integrate with yet.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _find_orvima_src() -> Path | None:
    candidates: list[Path] = []
    env = os.environ.get("ORVIMA_SRC")
    if env:
        candidates.append(Path(env))
    candidates.append(Path(__file__).resolve().parent.parent.parent / "orvima" / "src")
    candidates.append(Path.home() / "orvima" / "src")
    for candidate in candidates:
        if (candidate / "orvima" / "tools.py").is_file():
            return candidate
    return None


_orvima_src = _find_orvima_src()
if _orvima_src is not None and str(_orvima_src) not in sys.path:
    sys.path.insert(0, str(_orvima_src))
