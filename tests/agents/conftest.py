import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fakes import install


@pytest.fixture
def fake_env(monkeypatch):
    """Returns install(script, tools=None) -> (client, tools)."""

    def _install(script, tools=None):
        return install(monkeypatch, script, tools)

    return _install
