"""Unit tests for paths configuration."""

from ares.common.paths import ARES_ROOT, ARES_DATA_ROOT, ARES_CACHE_ROOT, MILESTONES_ROOT, ensure_dirs


def test_paths_defined():
    assert ARES_ROOT.exists()
    assert ARES_DATA_ROOT is not None
    assert ARES_CACHE_ROOT is not None
    assert MILESTONES_ROOT.exists()


def test_ensure_dirs():
    ensure_dirs()
    assert ARES_DATA_ROOT.exists()
    assert ARES_CACHE_ROOT.exists()
    assert MILESTONES_ROOT.exists()
