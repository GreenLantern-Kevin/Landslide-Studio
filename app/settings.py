"""Read-only configuration/status helpers; importing this module never changes jobs."""
import importlib.util
import json
import os
from pathlib import Path
from dotenv import dotenv_values
from app.model_files import base_model_ready
ROOT=Path(__file__).resolve().parents[1]


def llm_credentials(llm, root=None):
    """Read the current project .env on every use; never expose or cache key values."""
    name = str(llm.get('api_key_env') or 'SJTU_API_KEY')
    env_file = Path(root or ROOT) / '.env'
    try:
        values = dotenv_values(env_file, encoding='utf-8-sig', interpolate=False) if env_file.is_file() else {}
    except (OSError, UnicodeError) as exc:
        raise ValueError('无法读取工作台 .env，请检查文件编码和读取权限。') from exc
    if name in values:
        return (values[name] or '').strip(), 'project_env'
    return os.environ.get(name, '').strip(), 'process_env'

def config():
    def merge(a, b):
        for k, v in b.items():
            if isinstance(v, dict) and isinstance(a.get(k), dict):
                merge(a[k], v)
            else:
                a[k] = v
        return a
    cfg = json.loads((ROOT / 'config.example.json').read_text(encoding='utf-8-sig'))
    local = ROOT / 'config.local.json'
    if local.exists():
        merge(cfg, json.loads(local.read_text(encoding='utf-8-sig')))
    cfg['_root'] = str(ROOT)
    return cfg


def system():
    cfg = config()
    ai = all(importlib.util.find_spec(p) for p in ('torch', 'torchvision', 'diffusers', 'peft'))
    generation = {}
    for key, value in cfg['generation']['models'].items():
        ready = base_model_ready(ROOT/value['path'])
        generation[key] = {**value, 'base_ready': ready,
                           'download_incomplete': (ROOT/value['path']/'.download_manifest.json').is_file() and not ready,
                           'landslide_ready': (ROOT/value['lora_dir']/'Southwest_China_Landslide.safetensors').is_file(),
                           'terrain_ready': (ROOT/value['lora_dir']/'Southwest_China_Terrain.safetensors').is_file()}
    seg = cfg['segmentation']
    llm = cfg['llm']
    key, key_source = llm_credentials(llm, cfg['_root'])
    return {'root': str(ROOT), 'ai_installed': bool(ai), 'generation': generation,
            'segmentation': {**seg, 'rgb_ready': (ROOT/seg['rgb_weights']).is_file(),
                             'multimodal_ready': (ROOT/seg['multimodal_weights']).is_file(),
                             'repo_ready': (ROOT/seg['repo_path']).is_dir()},
            'llm': {'base_url': llm.get('base_url'), 'model': llm.get('model'),
                    'key_ready': bool(key), 'key_source': key_source,
                    'supports_vision': llm.get('supports_vision', False)}, 'max_upload_mb': cfg.get('max_upload_mb', 1024)}

