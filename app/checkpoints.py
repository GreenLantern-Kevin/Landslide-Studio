"""Restricted checkpoint loading and lightweight, explicit metadata inspection.

Some training checkpoints store a NumPy scalar for their best validation score.
Permit only NumPy's numeric scalar/dtype machinery in addition to PyTorch's
weights-only defaults; never import a class named by a checkpoint.
"""
from __future__ import annotations

from pathlib import Path
from threading import RLock

_LOAD_LOCK = RLock()


def _numpy_scalar_globals():
    import numpy as np
    from numpy.core.multiarray import scalar

    numeric_types = (np.bool_, np.int8, np.int16, np.int32, np.int64,
                     np.uint8, np.uint16, np.uint32, np.uint64,
                     np.float16, np.float32, np.float64, np.complex64, np.complex128)
    # NumPy 1.x and 2.x serialize the same scalar constructor under different
    # module names. These aliases point to our installed NumPy function only.
    return [(scalar, "numpy.core.multiarray.scalar"),
            (scalar, "numpy._core.multiarray.scalar"), np.dtype,
            *{type(np.dtype(kind)) for kind in numeric_types}]


def load_checkpoint(path: Path, *, metadata_only: bool = False):
    import torch

    allowed_static = {"numpy.core.multiarray.scalar", "numpy._core.multiarray.scalar", "numpy.dtype"}
    try:
        unsafe = set(torch.serialization.get_unsafe_globals_in_checkpoint(path))
        rejected = unsafe - allowed_static
        if rejected:
            raise ValueError("权重包含不支持的 Python 对象：" + ", ".join(sorted(rejected)))
        with _LOAD_LOCK, torch.serialization.safe_globals(_numpy_scalar_globals()):
            if metadata_only:
                from torch._subclasses.fake_tensor import FakeTensorMode
                with FakeTensorMode():
                    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            else:
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(
            "无法安全读取分割权重。平台保留 weights_only=True，仅兼容训练指标中的 NumPy 数值标量。"
            "若仍有其他 Python 对象，请在您信任的训练环境导出纯 model_state/state_dict 后放入。"
            f"错误：{type(exc).__name__}: {str(exc)[:1600]}"
        ) from exc
    if not isinstance(checkpoint, dict):
        raise ValueError("权重必须为张量 state_dict 或包含 model_state/state_dict 的字典。")
    return checkpoint


def checkpoint_state(checkpoint):
    import torch

    state = checkpoint.get("model_state", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state, dict) or not state or not all(isinstance(k, str) and torch.is_tensor(v) for k, v in state.items()):
        raise ValueError("权重中没有有效的模型参数字典，请使用仓库保存的 .pth/.pt checkpoint。")
    for prefix in ("module.", "_orig_mod."):
        if all(k.startswith(prefix) for k in state):
            state = {k[len(prefix):]: v for k, v in state.items()}
    return state


def checkpoint_metadata(checkpoint, state=None):
    """Read only scalar training settings and known output-head dimensions."""
    state = checkpoint_state(checkpoint) if state is None else state
    fields = {"model_name": ("model_name", "model"),
              "output_stride": ("output_stride",), "num_classes": ("num_classes",),
              "input_size": ("input_size", "crop_size"), "separable_conv": ("separable_conv",)}
    sources = [("checkpoint", checkpoint)]
    for key in ("config", "args", "opts", "hparams", "training_config"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            sources.append((key, value))
            if isinstance(value.get("segmentation"), dict):
                sources.append((key + ".segmentation", value["segmentation"]))
    recorded, provenance = {}, {}
    for target, names in fields.items():
        for source, values in sources:
            found = False
            for name in names:
                value = values.get(name)
                if target == "model_name":
                    valid = isinstance(value, str) and len(value) <= 128
                elif target == "separable_conv":
                    valid = isinstance(value, bool)
                else:
                    valid = isinstance(value, int) and not isinstance(value, bool) and value > 0
                if valid:
                    recorded[target], provenance[target] = value, source + "." + name
                    found = True
                    break
            if found:
                break
    inferred = {}
    # Known DeepLab/M2F final classifier keys; do not guess from arbitrary layers.
    heads = [state[name] for name in ("classifier.classifier.3.weight", "classifier.classifier.4.weight")
             if name in state and state[name].ndim == 4 and tuple(state[name].shape[2:]) == (1, 1)]
    class_counts = {int(tensor.shape[0]) for tensor in heads}
    if len(class_counts) == 1:
        inferred["num_classes"] = class_counts.pop()
    return {"recorded": recorded, "recorded_sources": provenance, "inferred": inferred,
            "tensor_count": len(state),
            "has_dem_backbone": any("dem_backbone" in key for key in state),
            "has_fhm": any("fhm" in key.lower() for key in state),
            "not_inferable_from_shapes": ["output_stride", "input_size", "dem_normalization", "channel_encoding"],
            "note": "output_stride 和训练裁剪大小不存于普通卷积权重形状；缺少明确元数据时使用用户配置，不猜测。"}


def resolve_checkpoint_config(config, metadata, overrides=None):
    """Recorded training settings first, with explicit per-run choices last."""
    effective = dict(config)
    sources = {name: "platform_config" for name in
               ("model_name", "output_stride", "num_classes", "input_size", "separable_conv")}
    if config.get("auto_match_checkpoint", True):
        for name, value in metadata.get("inferred", {}).items():
            effective[name], sources[name] = value, "tensor_shape"
        for name, value in metadata.get("recorded", {}).items():
            effective[name], sources[name] = value, metadata["recorded_sources"][name]
    for name in ("output_stride", "input_size", "tile_overlap"):
        if overrides and overrides.get(name) is not None:
            effective[name], sources[name] = overrides[name], "request"
    return effective, sources


def inspect_checkpoint(path: Path):
    checkpoint = load_checkpoint(Path(path), metadata_only=True)
    state = checkpoint_state(checkpoint)
    result = checkpoint_metadata(checkpoint, state)
    result.update(filename=Path(path).name, checkpoint_keys=list(checkpoint)[:20],
                  sample_shapes={key: list(value.shape) for key, value in list(state.items())[:12]})
    return result
