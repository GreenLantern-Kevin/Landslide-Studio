"""Inference adapter for M2F-DeepLab, with explicit legacy compatibility.

Uses the repository's actual network factory, never its training entry point.
All filenames returned below are relative to out_dir; the web layer adds URLs.
"""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path

from app.checkpoints import checkpoint_metadata, checkpoint_state, load_checkpoint, resolve_inference_config


REVIEWED_COMMIT = "e655ee310f5e9dfafceb33ca2afc92f214e14c29"
LEGACY_COMMIT = "149234e452cd39c5524b2c6e536b445a854a121e"
DEFAULT_MODEL = "m2f_deeplabv3plus_xception16"


def modern_backend(cfg):
    return cfg.get("backend") == "m2f_deeplab" or (
        not cfg.get("backend") and Path(cfg.get("repo_path", "")).name == "M2F-DeepLab")


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _load_network(repo: Path):
    """An isolated package name prevents collisions with other backends' network."""
    entry = repo / "network" / "__init__.py"
    if not entry.is_file():
        raise ValueError(f"未找到分割模型源码：{entry}。请安装后端或设置 segmentation.repo_path。")
    package = "_landslide_m2f_" + hashlib.sha256(str(repo.resolve()).encode()).hexdigest()[:12]
    if package not in sys.modules:
        spec = importlib.util.spec_from_file_location(package, entry, submodule_search_locations=[str(entry.parent)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[package] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            for key in list(sys.modules):
                if key == package or key.startswith(package + "."):
                    sys.modules.pop(key, None)
            raise
    return sys.modules[package]


def _checkpoint_state(checkpoint, torch):
    return checkpoint_state(checkpoint), checkpoint


def _inference_state(state, expected, metadata, model_name, modern):
    """Omit only the verified schema-2 training-only auxiliary head.

    Every inference tensor must match first. Keep the checkpoint on disk
    intact and retain strict=True; never tolerate arbitrary missing layers.
    """
    prefix = 'classifier.aux_classifier.'
    shapes = {'0.weight': (256, 728, 3, 3), '1.weight': (256,),
              '1.bias': (256,), '1.running_mean': (256,),
              '1.running_var': (256,), '1.num_batches_tracked': (),
              '3.weight': (2, 256, 1, 1), '3.bias': (2,)}
    auxiliary = {prefix + key: shape for key, shape in shapes.items()}
    extra = set(state) - set(expected)
    recorded = metadata.get('recorded', {})
    if (modern and metadata.get('schema') == 2
            and model_name == DEFAULT_MODEL and recorded.get('model_name') == DEFAULT_MODEL
            and recorded.get('output_stride') in (8, 16)
            and recorded.get('num_classes') == 2 and recorded.get('separable_conv') is False
            and extra == set(auxiliary) and not (set(expected) - set(state))
            and all(tuple(state[k].shape) == shape for k, shape in auxiliary.items())
            and all(state[k].shape == tensor.shape for k, tensor in expected.items())):
        return {k: v for k, v in state.items() if k not in auxiliary}, sorted(auxiliary)
    return state, []


def _build_model(cfg, weights, model_name, repo, torch):
    network = _load_network(repo)
    modern = bool(getattr(network.modeling, "MODEL_NAMES", ()))
    cfg = dict(cfg, backend="m2f_deeplab" if modern else "legacy")
    checkpoint = load_checkpoint(weights)
    state = checkpoint_state(checkpoint)
    metadata = checkpoint_metadata(checkpoint, state)
    # Release the checkpoint's optimizer/scheduler storage before allocating a
    # fresh inference network. The model tensors remain referenced by state.
    del checkpoint
    cfg, sources = resolve_inference_config(cfg, metadata, cfg.get("_request_overrides"), modern=modern)
    model_name = cfg.get("model_name", model_name)
    if int(cfg.get("output_stride", 8)) not in (8, 16):
        raise ValueError("output_stride 只能是 8 或 16，必须与训练时相同。")
    if not 2 <= int(cfg.get("num_classes", 2)) <= 256:
        raise ValueError("当前适配器使用多类别 softmax 输出，num_classes 必须为 2–256。")
    if modern and (int(cfg.get('num_classes', 2)) != 2 or int(cfg.get('landslide_class', 1)) != 1):
        raise ValueError("M2F-DeepLab 推理使用 2 类输出，0=背景、1=滑坡，请核对权重和类别设置。")
    recorded_name = metadata["recorded"].get("model_name")
    if recorded_name and recorded_name != model_name:
        raise ValueError(f"模型配置 {model_name} 与权重记录 {recorded_name} 不一致，请修改模型名称。")
    allowed = set(network.modeling.MODEL_NAMES) if modern else {name for name in vars(network.modeling) if name.startswith(("m2f_deeplabv3plus_", "me_deeplabv3plus_", "single_deeplabv3plus_", "deeplabv3plus_"))}
    if model_name not in allowed:
        raise ValueError(f"仓库中不存在模型 {model_name}。可用模型：{', '.join(sorted(allowed))}")
    model = getattr(network.modeling, model_name)(
        num_classes=int(cfg.get("num_classes", 2)),
        output_stride=int(cfg.get("output_stride", 8)),
        pretrained_backbone=False,
    )
    if cfg.get("separable_conv", False):
        network.convert_to_separable_conv(model.classifier)
    expected = model.state_dict()
    state, omitted_auxiliary = _inference_state(state, expected, metadata, model_name, modern)
    metadata['inference_loading'] = {'strict': True, 'loaded_tensors': len(state),
                                   'omitted_training_auxiliary': omitted_auxiliary}
    missing = sorted(set(expected) - set(state))
    extra = sorted(set(state) - set(expected))
    shape_errors = [f"{k}: 权重 {tuple(state[k].shape)} / 模型 {tuple(expected[k].shape)}" for k in set(expected) & set(state) if expected[k].shape != state[k].shape]
    if missing or extra or shape_errors:
        raise ValueError(
            "权重与模型架构不匹配，已停止推理。请核对 model_name、num_classes、separable_conv 以及训练时的代码版本。"
            + ("新版 M2F-DeepLab 仅兼容已确认结构的 schema=2 训练辅助头；其他多余、缺失或形状不符的参数仍会拒绝，请核对训练代码版本。" if modern else "")
            +
            f"缺失 {len(missing)} 项 {missing[:4]}；多出 {len(extra)} 项 {extra[:4]}；形状差异 {shape_errors[:4]}"
        )
    model.load_state_dict(state, strict=True)
    model.studio_inference_config = cfg
    model.studio_checkpoint_metadata = metadata
    model.studio_parameter_sources = sources
    return model.eval()


def _read_image(path, rgb, max_pixels):
    import numpy as np
    from PIL import Image

    # TIFF must be opened with rasterio so georeferencing is not silently discarded.
    if path.suffix.lower() in (".tif", ".tiff"):
        try:
            import rasterio
        except ImportError as exc:
            raise ValueError("读取 TIFF/GeoTIFF 需要 rasterio。请先运行依赖安装脚本；普通 PNG/JPEG 不需要此依赖。") from exc
        with rasterio.open(path) as ds:
            if ds.width * ds.height > max_pixels:
                raise ValueError(f"影像超过当前 {max_pixels:,} 像素上限，请先裁剪研究区或调高配置。")
            if rgb and ds.count < 3:
                raise ValueError("RGB TIFF 至少需要 3 个波段，默认按第 1/2/3 波段读取红/绿/蓝。")
            indexes = [1, 2, 3] if rgb else list(range(1, ds.count + 1))
            if not rgb and len(indexes) not in (1, 3):
                raise ValueError("DEM 只支持单波段高程或预处理后的三波段地形图。")
            data = ds.read(indexes)
            valid = (ds.read_masks(indexes) > 0).all(axis=0)
            profile = dict(ds.profile)
            geo = {"crs": ds.crs, "transform": ds.transform, "profile": profile}
            arr = data.transpose(1, 2, 0)
            if arr.shape[2] == 1:
                arr = arr[:, :, 0]
            valid &= np.isfinite(arr).all(axis=2) if arr.ndim == 3 else np.isfinite(arr)
            return arr, valid, geo
    if not rgb:
        with Path(path).open("rb") as stream:
            header = stream.read(26)
        if header.startswith(b'\x89PNG\r\n\x1a\n') and len(header) > 24 and header[24] != 8:
            raise ValueError("DEM PNG 必须是训练所用的 8 位三通道地形图，不能隐式降位深。")
    with Image.open(path) as im:
        if im.width * im.height > max_pixels:
            raise ValueError(f"影像超过当前 {max_pixels:,} 像素上限，请先裁剪研究区。")
        if rgb:
            if im.mode not in ("RGB", "RGBA", "P", "CMYK", "L"):
                raise ValueError("RGB 影像格式不支持，请导出为 8 位 RGB PNG/JPEG。")
            alpha = np.asarray(im.getchannel("A")) > 0 if im.mode == "RGBA" else None
            arr = np.asarray(im.convert("RGB")).copy()
        else:
            arr = np.asarray(im).copy()
            alpha = None
    valid = np.isfinite(arr).all(axis=2) if arr.ndim == 3 else np.isfinite(arr)
    if alpha is not None:
        valid &= alpha
    return arr, valid, None


def _align_dem(rgb_shape, rgb_geo, dem, dem_valid, dem_geo, params):
    import numpy as np

    rgb_has_geo = bool(rgb_geo and rgb_geo["crs"])
    dem_has_geo = bool(dem_geo and dem_geo["crs"])
    shape_ok = tuple(dem.shape[:2]) == tuple(rgb_shape[:2])
    if rgb_has_geo and dem_has_geo:
        same = shape_ok and rgb_geo["crs"] == dem_geo["crs"] and rgb_geo["transform"].almost_equals(dem_geo["transform"], precision=1e-8)
        if same:
            return dem, dem_valid, "已校验 RGB 与 DEM 的 CRS、仿射变换、尺寸一致"
        if not params.get("reproject_dem", False):
            raise ValueError("RGB 与 DEM 的地理网格不一致（CRS、像元位置或尺寸）。请配准后上传，或开启“按 RGB 网格重投影 DEM”。仅尺寸相同不能证明已对齐。")
        from rasterio.warp import reproject, Resampling
        h, w = rgb_shape[:2]
        source = dem[:, :, None] if dem.ndim == 2 else dem
        result = np.full((h, w, source.shape[2]), np.nan, dtype=np.float32)
        for i in range(source.shape[2]):
            band = np.where(dem_valid, source[:, :, i], np.nan).astype(np.float32)
            reproject(band, result[:, :, i], src_transform=dem_geo["transform"], src_crs=dem_geo["crs"], dst_transform=rgb_geo["transform"], dst_crs=rgb_geo["crs"], src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.bilinear)
        valid = np.isfinite(result).all(axis=2)
        if not valid.any():
            raise ValueError("DEM 重投影后与 RGB 无有效重叠，请检查影像范围和坐标系。")
        return (result[:, :, 0] if dem.ndim == 2 else result), valid, "DEM 已按 RGB 的地理网格重投影（双线性）"
    if not shape_ok:
        raise ValueError("缺少可用地理坐标信息，且 RGB/DEM 尺寸不同。请先在 GIS 中配准并导出相同网格，平台不会直接拉伸 DEM 冒充配准。")
    if not params.get("alignment_confirmed", False):
        raise ValueError("RGB 或 DEM 缺少地理坐标，无法自动验证配准。请确认两者覆盖相同范围、方向、像元位置后勾选“我已确认 RGB 与 DEM 逐像元配准”。")
    return dem, dem_valid, "用户确认已逐像元配准；文件缺少完整地理坐标，平台无法独立验证"


def _normalize_channel(channel, valid):
    import numpy as np
    values = channel[valid]
    low, high = float(values.min()), float(values.max())
    if high - low < 1e-5:
        return np.zeros(channel.shape, dtype=np.uint8)
    return np.clip((channel - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)


def _dem_features(dem, valid, mode):
    """Matches prepare_dem.py's elevation/Sobel magnitude/aspect RGB ordering."""
    import cv2
    import numpy as np
    if not valid.any():
        raise ValueError("DEM 没有有效像素。")
    if mode == "prepared_rgb":
        if dem.ndim != 3 or dem.shape[2] != 3:
            raise ValueError("prepared_rgb 模式要求与训练时文件 RGB 解码通道顺序、像素编码一致的三通道地形图。")
        values = dem[valid]
        if values.min() < 0 or values.max() > 255:
            raise ValueError("预处理地形图应在 0–255 范围内，请使用训练时的预处理格式。")
        return np.where(valid[:, :, None], dem, 0).astype(np.uint8)
    if mode != "raw_elevation":
        raise ValueError("dem_input 必须为 raw_elevation 或 prepared_rgb。")
    if dem.ndim != 2:
        raise ValueError("原始 DEM 模式只接收单通道高程。三通道地形图请切换 dem_input=prepared_rgb。")
    # Invalid areas are excluded from statistics and prediction. Filling avoids NaN propagation.
    elevation = np.where(valid, dem, np.median(dem[valid])).astype(np.float32)
    dx = cv2.Sobel(elevation, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(elevation, cv2.CV_32F, 0, 1, ksize=3)
    channels = (elevation, np.sqrt(dx ** 2 + dy ** 2), np.arctan2(dy, dx))
    return np.stack([_normalize_channel(x, valid) for x in channels], axis=-1)


def _starts(length, size, overlap):
    if length <= size:
        return [0]
    starts = list(range(0, length - size + 1, size - overlap))
    if starts[-1] != length - size:
        starts.append(length - size)
    return starts


def _predict_probability(model, rgb, dem_features, cfg, params, device, torch, progress):
    import cv2
    import numpy as np
    from PIL import Image
    h, w = rgb.shape[:2]
    strategy = params.get("strategy", cfg.get('strategy', 'native'))
    if strategy == 'auto':
        strategy = 'resize' if params.get('input_size') is not None or cfg.get('eval_resize', 256) else 'native'
    if strategy not in ('native', 'tiles', 'resize'):
        raise ValueError('推理策略必须为 auto、native、tiles 或 resize。')
    size = int(params['input_size'] if params.get('input_size') is not None else cfg.get('input_size', 256))
    if strategy != 'native' and (size < 64 or size > 1024):
        raise ValueError("窗口/缩放尺寸应为 64–1024 之间的整数。")
    overlap = int(params['tile_overlap'] if params.get('tile_overlap') is not None else cfg.get('tile_overlap', 64))
    if strategy == 'tiles' and (overlap < 0 or overlap >= size):
        raise ValueError("分块重叠像素应大于等于 0 且小于推理尺寸。")
    rgb_mean = np.asarray(cfg.get("rgb_mean", [0.485, 0.456, 0.406]), dtype=np.float32)[None, None, :]
    rgb_std = np.asarray(cfg.get("rgb_std", [0.229, 0.224, 0.225]), dtype=np.float32)[None, None, :]
    if rgb_mean.shape != (1, 1, 3) or rgb_std.shape != (1, 1, 3) or (rgb_std <= 0).any():
        raise ValueError("rgb_mean/std 必须各包含 3 个值，std 必须为正。")
    multimodal = hasattr(model.backbone, "dem_backbone")
    explicit_dem = multimodal and "has_dem" in inspect.signature(model.forward).parameters
    if dem_features is not None and not multimodal:
        raise ValueError("当前权重对应 RGB 单流模型，不能接收 DEM。请选择 M2F 双流模型与匹配的权重。")
    class_id = int(cfg.get("landslide_class", 1))
    dem_norm = cfg.get("dem_normalization", "imagenet")
    if dem_norm not in ("imagenet", "unit"):
        raise ValueError("dem_normalization 必须为 imagenet 或 unit，并与训练时一致。")
    if modern_backend(cfg) and (dem_norm != 'imagenet' or not np.array_equal(rgb_mean, np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[None, None, :]) or not np.array_equal(rgb_std, np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[None, None, :])):
        raise ValueError("M2F-DeepLab 的 RGB 和三通道 DEM 必须使用仓库一致的 ImageNet 标准化。")
    missing_mode = cfg.get("missing_dem_mode", "zeros")
    if missing_mode not in ("zeros", "normalized_black"):
        raise ValueError("missing_dem_mode 必须为 zeros 或 normalized_black。")

    def one(rgb_crop, dem_crop):
        x = ((rgb_crop.astype(np.float32) / 255.0 - rgb_mean) / rgb_std).transpose(2, 0, 1)
        x = torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0).to(device)
        if multimodal:
            if dem_crop is None:
                d = torch.zeros_like(x)
                if missing_mode == "normalized_black" and not explicit_dem:
                    d = torch.from_numpy(np.broadcast_to(-rgb_mean / rgb_std, rgb_crop.shape).transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
            else:
                d = dem_crop.astype(np.float32) / 255.0
                if dem_norm == "imagenet":
                    d = (d - rgb_mean) / rgb_std
                d = torch.from_numpy(np.ascontiguousarray(d.transpose(2, 0, 1))).unsqueeze(0).to(device)
            if explicit_dem:
                out = model(x, d, has_dem=torch.tensor([dem_crop is not None], dtype=torch.bool, device=device))
            else:
                out = model(x, d)
        else:
            out = model(x)
        logits = out["seg"] if isinstance(out, dict) else out
        if logits.ndim != 4 or not 0 <= class_id < logits.shape[1]:
            raise ValueError("模型输出与 landslide_class 配置不一致。")
        probability = torch.softmax(logits.float(), dim=1)[0, class_id].cpu().numpy()
        if not np.isfinite(probability).all():
            raise ValueError("模型输出出现 NaN/Inf，请检查权重和预处理配置。")
        return probability

    with torch.inference_mode():
        if strategy == 'native':
            probability = one(rgb, dem_features)
            progress(85, '按输入原始尺寸完成推理，未缩放或分块')
            return probability, 1, strategy
        if strategy == "resize":
            r = np.asarray(Image.fromarray(rgb).resize((size, size), Image.Resampling.BILINEAR))
            d = np.asarray(Image.fromarray(dem_features).resize((size, size), Image.Resampling.BILINEAR)) if dem_features is not None else None
            grid_probability = one(r, d)
            if modern_backend(cfg):
                # Nearest restore commutes with thresholding, matching upstream's
                # network-grid argmax followed by nearest-neighbor mask export.
                probability = np.array(Image.fromarray(grid_probability).resize((w, h), Image.Resampling.NEAREST), copy=True)
            else:
                probability = cv2.resize(grid_probability, (w, h), interpolation=cv2.INTER_LINEAR)
            progress(85, "整图推理完成，恢复原始尺寸")
            return probability, 1, strategy
        ys, xs = _starts(h, size, overlap), _starts(w, size, overlap)
        count = len(ys) * len(xs)
        accumulated = np.zeros((h, w), dtype=np.float32)
        weight_sum = np.zeros((h, w), dtype=np.float32)
        # Smooth positive weights avoid hard seams and zero coverage at outer edges.
        taper = np.maximum(np.hanning(size), 0.05).astype(np.float32)
        window = np.outer(taper, taper)
        for index, (y, x) in enumerate((y, x) for y in ys for x in xs):
            bh, bw = min(size, h - y), min(size, w - x)
            r = rgb[y:y + bh, x:x + bw]
            padding = ((0, size - bh), (0, size - bw), (0, 0))
            r = np.pad(r, padding, mode="edge")
            d = np.pad(dem_features[y:y + bh, x:x + bw], padding, mode="edge") if dem_features is not None else None
            p = one(r, d)[:bh, :bw]
            weights = window[:bh, :bw]
            accumulated[y:y + bh, x:x + bw] += p * weights
            weight_sum[y:y + bh, x:x + bw] += weights
            progress(20 + int(65 * (index + 1) / count), f"滑窗推理 {index + 1}/{count}")
        return accumulated / np.maximum(weight_sum, 1e-8), count, strategy


def run_segmentation(rgb_path: Path, dem_path: Path | None, out_dir: Path, params: dict, config: dict, progress) -> dict:
    """Return mask/overlay/probability/metrics filenames and honest descriptive metrics."""
    cfg = dict(config.get("segmentation", {}))
    cfg["_request_overrides"] = {name: params[name] for name in ("output_stride", "input_size", "tile_overlap")
                                 if params.get(name) is not None}
    cfg.update(cfg["_request_overrides"])
    root = Path(config.get("_root", "."))
    rgb_path, out_dir = Path(rgb_path), Path(out_dir)
    dem_path = Path(dem_path) if dem_path else None
    has_rgb_weights = bool(cfg.get("rgb_weights"))
    weights_value = cfg.get("multimodal_weights") if dem_path or not has_rgb_weights else cfg.get("rgb_weights")
    if not weights_value:
        raise ValueError("尚未配置分割权重。请把训练完成的 .pth/.pt 放入 models/segmentation，并在设置中填写 multimodal_weights（或 rgb_weights）；平台不会使用随机模型生成结果。")
    weights = _resolve(root, weights_value)
    if not weights.is_file():
        raise ValueError(f"未找到分割权重：{weights}。请放入真实训练权重后重试。")
    if not rgb_path.is_file() or (dem_path and not dem_path.is_file()):
        raise ValueError("上传的 RGB 或 DEM 文件不存在，请重新上传。")
    model_name = cfg.get("model_name", DEFAULT_MODEL)
    if not dem_path and has_rgb_weights:
        model_name = cfg.get("rgb_model_name") or model_name
    cfg["model_name"] = model_name
    repo = _resolve(root, cfg.get("repo_path", "vendor/M2F-DeepLab"))
    try:
        import numpy as np
        import cv2
        import torch
        from PIL import Image
    except ImportError as exc:
        raise ValueError("分割推理依赖未安装，请运行安装脚本（需要 torch、torchvision、numpy、Pillow、opencv-python-headless）。") from exc
    progress(3, "检查影像与地理配准信息")
    max_pixels = int(cfg.get("max_image_pixels", 16000000))
    rgb, valid, geo = _read_image(rgb_path, True, max_pixels)
    if rgb.dtype != np.uint8:
        raise ValueError("RGB TIFF 不是 8 位影像。请按训练时相同方式转换为 uint8 RGB，避免平台自动拉伸改变模型输入分布。")
    dem_features, alignment = None, "仅 RGB 推理"
    dem_input = params.get("dem_input", cfg.get("dem_input", "prepared_rgb")) if dem_path else None
    if dem_path and modern_backend(cfg) and dem_input != 'prepared_rgb':
        raise ValueError("新版 M2F-DeepLab 仅接收预处理的三通道高程/坡度/坡向地形图，请先按训练方式处理原始 DEM。")
    if dem_path:
        dem, dem_valid, dem_geo = _read_image(dem_path, False, max_pixels)
        if modern_backend(cfg) and (dem.dtype != np.uint8 or dem.ndim != 3 or dem.shape[2] != 3):
            raise ValueError("M2F-DeepLab 的 DEM 必须为 uint8 三通道高程/坡度/坡向地形图，不进行自动降位深或灰度复制。")
        dem, dem_valid, alignment = _align_dem(rgb.shape, geo, dem, dem_valid, dem_geo, params)
        valid &= dem_valid
        dem_features = _dem_features(dem, dem_valid, dem_input)
    if not valid.any():
        raise ValueError("输入影像没有共同有效像素，无法推理。")
    requested_device = params.get("device", cfg.get("device", "auto"))
    if requested_device not in ("auto", "cpu", "cuda"):
        raise ValueError("device 必须为 auto、cpu 或 cuda。")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise ValueError("当前 PyTorch 无可用 CUDA，请安装正确的 GPU 版依赖或选择 CPU。")
    device = torch.device("cuda" if requested_device == "auto" and torch.cuda.is_available() else "cpu" if requested_device == "auto" else requested_device)
    threshold = float(params.get("threshold", 0.5))
    alpha = float(params.get("overlay_alpha", 0.45))
    min_area = int(params.get("min_area", 0))
    if not 0 < threshold < 1 or not 0 <= alpha <= 1 or min_area < 0:
        raise ValueError("阈值需在 0 与 1 之间，叠加透明度需为 0–1，最小斑块面积不得为负数。")
    progress(10, f"加载 {model_name} 并逐项校验权重")
    model = _build_model(cfg, weights, model_name, repo, torch)
    cfg = getattr(model, "studio_inference_config", cfg)
    model_name = cfg.get("model_name", model_name)
    parameter_sources = getattr(model, "studio_parameter_sources", {})
    checkpoint_info = getattr(model, "studio_checkpoint_metadata", {})
    try:
        model.to(device)
        probability, tiles, strategy = _predict_probability(model, rgb, dem_features, cfg, params, device, torch, progress)
    except torch.cuda.OutOfMemoryError as exc:
        raise ValueError('显存不足：请在分割页面改用“滑窗推理”，窗口设为256、重叠64，并关闭占用显存的其他程序。') from exc
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    probability[~valid] = np.nan
    # Binary argmax ties select background (class 0) in upstream inference.
    positive_threshold = probability > threshold if modern_backend(cfg) else probability >= threshold
    binary = (positive_threshold & valid).astype(np.uint8)
    components, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if min_area:
        kept = np.zeros(components, dtype=np.uint8)
        kept[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
        binary = kept[labels]
        components, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    positive = binary.astype(bool)
    pixels, total = int(positive.sum()), int(valid.sum())
    area_m2 = None
    if geo and geo["crs"] and geo["crs"].is_projected:
        try:
            _, factor = geo["crs"].linear_units_factor
            t = geo["transform"]
            area_m2 = float(pixels * abs(t.a * t.e - t.b * t.d) * factor ** 2)
        except (AttributeError, ValueError):
            pass
    warnings = ["输出为模型预测，概率未经校准；没有真值标签，因此不报告准确率、IoU 或 F1。"]
    omitted_auxiliary = checkpoint_info.get('inference_loading', {}).get('omitted_training_auxiliary', [])
    if omitted_auxiliary:
        warnings.append(f"已兼容 schema=2 权重：仅在内存中排除 {len(omitted_auxiliary)} 项不参与 seg/edge 推理的训练辅助头参数，全部推理参数严格加载；原权重文件未修改。")
    if dem_path:
        if dem_input == "prepared_rgb":
            warnings.append("已读取训练所用格式的三通道地形图，未重新计算地形特征或交换通道；请确保文件像素编码与训练一致。")
        else:
            warnings.append("原始 DEM 已派生高程/梯度幅值/坡向通道；梯度幅值不是物理坡度角，仅适用于训练采用相同派生方法的权重。")
    elif modern_backend(cfg) and model_name.startswith(("m2f_", "me_")):
        route = '零地形特征' if model_name.endswith('_b') else 'FHM 特征幻觉'
        warnings.append(f"缺失 DEM：显式 has_dem=False，使用{route}路径；效果取决于对应权重的训练。")
    elif model_name.startswith(("m2f_", "me_")):
        if cfg.get("missing_dem_mode", "zeros") == "zeros":
            warnings.append("缺失 DEM 时使用仓库的特征幻觉路径；效果取决于权重是否充分训练过缺失模态样本。")
        else:
            warnings.append("缺失 DEM 时使用经标准化的黑色占位图以复现训练数据输入；这会走 DEM 编码器，效果需验证。")
    if strategy == "resize":
        warnings.append("整图按所选评估尺寸缩放后恢复原图；新版掩膜采用最近邻恢复，与原仓库的网络网格判别一致。改变尺寸或滑窗策略会影响结果。")
    metrics = {
        "model": model_name, "weights": weights.name, "reviewed_repo_commit": REVIEWED_COMMIT if modern_backend(cfg) else LEGACY_COMMIT,
        "backend": cfg.get('backend', 'legacy'), "repo_path": cfg.get('repo_path'),
        "has_dem": dem_path is not None,
        "dem_route": ('real_dem' if dem_path else 'zero_features' if model_name.endswith('_b') else 'fhm') if model_name.startswith(('m2f_', 'me_')) else 'rgb',
        "requested_strategy": params.get('strategy', cfg.get('strategy', 'native')),
        "eval_resize": cfg.get('eval_resize'),
        "probability_restore": 'nearest' if modern_backend(cfg) and strategy == 'resize' else 'native_or_blended',
        "input_mode": "RGB+DEM" if dem_path else "RGB", "device": str(device),
        "output_stride": int(cfg.get("output_stride", 8)),
        "input_size": None if strategy == 'native' else int(cfg.get("input_size", 256)),
        "configured_input_size": int(cfg.get("input_size", 256)),
        "num_classes": int(cfg.get("num_classes", 2)),
        "parameter_sources": parameter_sources, "checkpoint_metadata": checkpoint_info,
        "width": int(rgb.shape[1]), "height": int(rgb.shape[0]), "valid_pixels": total,
        "landslide_pixels": pixels, "landslide_ratio": pixels / total,
        "landslide_percent": 100 * pixels / total, "component_count": components - 1,
        "largest_component_pixels": int(stats[1:, cv2.CC_STAT_AREA].max()) if components > 1 else 0,
        "mean_landslide_probability": float(probability[positive].mean()) if pixels else None,
        "area_m2": area_m2, "threshold": threshold, "min_area": min_area,
        "strategy": strategy, "tiles": tiles, "alignment": alignment,
        "dem_input": dem_input, "dem_normalization": cfg.get("dem_normalization", "imagenet"), "warnings": warnings,
        "mask_encoding": "mask.png: 0=背景/无效像素, 255=滑坡；valid_mask.png: 255=有效像素",
    }
    progress(90, "保存掩膜、概率、叠加图和统计信息")
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(out_dir / "input_rgb.png")
    Image.fromarray(binary * 255).save(out_dir / "mask.png")
    Image.fromarray(valid.astype(np.uint8) * 255).save(out_dir / "valid_mask.png")
    overlay = rgb.copy()
    overlay[positive] = (rgb[positive] * (1 - alpha) + np.array([238, 104, 69]) * alpha).astype(np.uint8)
    Image.fromarray(overlay).save(out_dir / "overlay.png")
    np.save(out_dir / "probability.npy", probability.astype(np.float32))
    Image.fromarray((np.nan_to_num(probability) * 255).astype(np.uint8)).save(out_dir / "probability.png")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    files = [
        {"name": "input_rgb.png", "label": "原始 RGB", "kind": "image"},
        {"name": "overlay.png", "label": "滑坡叠加图", "kind": "image"},
        {"name": "mask.png", "label": "二值掩膜", "kind": "image"},
        {"name": "valid_mask.png", "label": "有效像素掩膜", "kind": "image"},
        {"name": "probability.png", "label": "概率预览", "kind": "image"},
        {"name": "probability.npy", "label": "原始概率 NPY", "kind": "data"},
        {"name": "metrics.json", "label": "统计 JSON", "kind": "data"},
    ]
    if geo and geo["crs"]:
        import rasterio
        profile = dict(geo["profile"])
        profile.update(count=1, dtype="uint8", nodata=255, compress="deflate")
        # Build a single-band profile without RGB-specific options.
        for field in ("photometric", "interleave"):
            profile.pop(field, None)
        geodata = binary.copy()
        geodata[~valid] = 255
        with rasterio.open(out_dir / "mask_geo.tif", "w", **profile) as ds:
            ds.write(geodata, 1)
        files.append({"name": "mask_geo.tif", "label": "地理参考掩膜 GeoTIFF", "kind": "data"})
    progress(100, "分割推理完成")
    summary = f"已完成 {metrics['input_mode']} 分割，预测滑坡像素占有效区域 {metrics['landslide_percent']:.2f}%，共 {metrics['component_count']} 个连通斑块。"
    return {"files": files, "metrics": metrics, "warnings": warnings, "summary": summary}
