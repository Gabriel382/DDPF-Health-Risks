"""Basic import tests for the DDPF package."""

import ddpf


def test_ddpf_import() -> None:
    """The package should be importable after installation."""
    assert ddpf.__version__ == "0.1.0"
