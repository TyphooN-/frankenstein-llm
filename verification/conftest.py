"""Keep synthetic performance observations out of live qualification evidence."""
import pytest


@pytest.fixture(autouse=True)
def isolate_qualification_performance(tmp_path, monkeypatch):
    monkeypatch.setenv('QUALIFICATION_PERFORMANCE_PATH',
                       str(tmp_path / 'test-observations.performance.jsonl'))
