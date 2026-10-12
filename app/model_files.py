"""Lightweight local model readiness checks; no network or weight deserialization."""
import json
from pathlib import Path


def base_model_ready(base: Path):
    if not (base / 'model_index.json').is_file():
        return False
    manifest = base / '.download_manifest.json'
    if not manifest.exists():
        # User-supplied pipelines are validated in detail by Diffusers during loading.
        return True
    try:
        data = json.loads(manifest.read_text(encoding='utf-8'))
        return bool(data.get('complete') and data.get('files')) and all(
            (base / item['path']).is_file() and (base / item['path']).stat().st_size == item['bytes']
            for item in data['files'])
    except (OSError, ValueError, KeyError, TypeError):
        return False


def model_variant(base: Path, spec: dict):
    if 'variant' in spec:
        return spec['variant'] or None
    # Selected downloads use fp16 filenames; standard full-precision directories still work.
    if (base / 'unet/diffusion_pytorch_model.fp16.safetensors').is_file() and not (
            base / 'unet/diffusion_pytorch_model.safetensors').is_file():
        return 'fp16'
    return None
