"""The explicit MLX backend must serve on Linux (Omarchy runs MLX through Vulkan), and the process footprint
must read a real value there instead of None."""

import sys
from types import SimpleNamespace

from tensorfold.cli import _backend
from tensorfold.server import memory_budget


def _family(*, load: bool = True, cuda: bool = False) -> SimpleNamespace:
    package = SimpleNamespace()
    if load:
        package.load = lambda *a, **k: None
    if cuda:
        package.cuda_engine = lambda *a, **k: None
    return SimpleNamespace(package=package, title="family", lanes=False)


def test_explicit_mlx_serves_on_linux(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    assert _backend("mlx", _family()) == "mlx"


def test_auto_policy_unchanged_on_linux(monkeypatch):
    """Auto keeps its CUDA default off macOS; the Vulkan path is asked for explicitly."""

    monkeypatch.setattr(sys, "platform", "linux")
    assert _backend("auto", _family(load=False, cuda=True)) == "cuda"


def test_footprint_reads_a_real_value_on_linux(monkeypatch):
    import ctypes

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(ctypes, "CDLL", _missing_cdll)
    monkeypatch.setattr("builtins.open", lambda *a, **k: _StatusFile())
    assert memory_budget.process_footprint() == 3 * 1024**2


class _StatusFile:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(["VmRSS:\t3072 kB\n"])


def _missing_cdll(*args, **kwargs):
    raise OSError("no libSystem on Linux")
