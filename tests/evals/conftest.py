import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "agents"))

from fakes import install


@pytest.fixture
def fake_env(monkeypatch):
    def _install(script, tools=None):
        return install(monkeypatch, script, tools)

    return _install
