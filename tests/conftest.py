"""Shared pytest configuration.

Skips tests marked ``live`` unless the run selects them with ``-m live``, so
the default suite never needs keys, the network, or the real Chroma store.
"""

from __future__ import annotations

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Mark every ``live`` test as skipped unless ``-m`` mentions ``live``."""
    if "live" in (config.getoption("markexpr") or ""):
        return
    skip_live = pytest.mark.skip(reason="live test: run with -m live")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
