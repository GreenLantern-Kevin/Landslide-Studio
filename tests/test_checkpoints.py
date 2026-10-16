"""Regression tests for safe NumPy metric compatibility and honest inference."""
from pathlib import Path

import numpy as np
import pytest
import torch

from app.checkpoints import (checkpoint_metadata, inspect_checkpoint, load_checkpoint,
                             resolve_checkpoint_config)


def _untrusted_constructor(marker):
    Path(marker).write_text("executed", encoding="utf-8")
    return {}


class _UntrustedObject:
    def __init__(self, marker):
        self.marker = str(marker)

    def __reduce__(self):
        return _untrusted_constructor, (self.marker,)


def test_numpy_training_score_loads_without_disabling_weights_only(tmp_path, monkeypatch):
    path = tmp_path / "trained.pth"
    torch.save({"model_state": {"weight": torch.ones(2, 3)}, "best_score": np.float64(0.84)}, path)
    original_load = torch.load
    observed = []

    def tracked_load(*args, **kwargs):
        observed.append(kwargs.get("weights_only"))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", tracked_load)
    result = load_checkpoint(path)
    assert result["best_score"] == np.float64(0.84)
    assert observed == [True]
    # The compatibility context must not persistently weaken future loads.
    assert "numpy.dtype" in torch.serialization.get_unsafe_globals_in_checkpoint(path)


def test_unknown_checkpoint_object_is_rejected_before_execution(tmp_path):
    path, marker = tmp_path / "object.pth", tmp_path / "executed.txt"
    torch.save({"model_state": {"weight": torch.ones(1)}, "object": _UntrustedObject(marker)}, path)
    with pytest.raises(ValueError, match="不支持的 Python 对象"):
        load_checkpoint(path)
    assert not marker.exists()


def test_object_dtype_is_not_on_numeric_allowlist(tmp_path):
    path = tmp_path / "object_dtype.pth"
    torch.save({"model_state": {"weight": torch.ones(1)}, "dtype": np.dtype(object)}, path)
    with pytest.raises(ValueError, match="无法安全读取"):
        load_checkpoint(path)


def test_metadata_inspection_distinguishes_recorded_and_inferred(tmp_path):
    path = tmp_path / "weight.pth"
    torch.save({"model_state": {"classifier.classifier.3.weight": torch.ones(2, 32, 1, 1)},
                "model_name": "m2f_deeplabv3plus_xception16", "best_score": np.float32(0.8)}, path)
    info = inspect_checkpoint(path)
    assert info["recorded"] == {"model_name": "m2f_deeplabv3plus_xception16"}
    assert info["inferred"] == {"num_classes": 2}
    assert "output_stride" not in info["recorded"]
    assert "input_size" not in info["inferred"]
    assert info["tensor_count"] == 1


def test_explicit_request_has_priority_over_metadata():
    ckpt = {"model_state": {"weight": torch.ones(1)},
            "config": {"output_stride": 8, "crop_size": 256}}
    info = checkpoint_metadata(ckpt)
    effective, sources = resolve_checkpoint_config({"output_stride": 8, "input_size": 320}, info,
                                                   {"output_stride": 16, "input_size": 512})
    assert effective["output_stride"] == 16
    assert effective["input_size"] == 512
    assert sources["output_stride"] == sources["input_size"] == "request"


def test_missing_metadata_never_guesses_stride_or_size():
    info = checkpoint_metadata({"weight": torch.ones(1)})
    effective, sources = resolve_checkpoint_config({"output_stride": 16, "input_size": 256}, info)
    assert effective["output_stride"] == 16 and effective["input_size"] == 256
    assert sources["output_stride"] == sources["input_size"] == "platform_config"
