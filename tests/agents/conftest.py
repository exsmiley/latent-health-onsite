import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fakes import install


@pytest.fixture
def fake_env(monkeypatch):
    """Returns install(script, tools=None, pre_retrieve=False) -> (client, tools)."""

    def _install(script, tools=None, pre_retrieve=False):
        return install(monkeypatch, script, tools, pre_retrieve)

    return _install
