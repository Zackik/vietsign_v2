import sys
import types

from vietsign.utils.device import select_device


def test_select_cpu_when_requested() -> None:
    assert select_device("cpu").type == "cpu"


def test_select_cpu_when_torch_missing(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)

    assert select_device().type == "cpu"


def test_device_priority_cuda_before_mps(monkeypatch) -> None:
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: True),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: True)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert select_device().type == "cuda"


def test_mps_used_when_cuda_unavailable(monkeypatch) -> None:
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: True)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert select_device().type == "mps"

