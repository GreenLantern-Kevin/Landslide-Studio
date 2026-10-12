"""Read only selected kohya metadata from local safetensors headers (no tensor loading)."""
import argparse
import json
from pathlib import Path
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ('ss_base_model_version', 'ss_sd_model_name', 'ss_v2', 'ss_v_parameterization',
          'ss_resolution', 'ss_network_module', 'ss_network_dim', 'ss_network_alpha')
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('file', nargs='?', help='Optional LoRA file, otherwise scan the three standard folders')
args = parser.parse_args()
files = [Path(args.file)] if args.file else sorted((ROOT / 'models/lora').glob('*/*.safetensors'))
if not files:
    print('No LoRA files found in models/lora/{sd15,sd21,sdxl}.')
for path in files:
    with safe_open(str(path), framework='pt', device='cpu') as stream:
        metadata = stream.metadata() or {}
        print(json.dumps({'file': str(path), 'tensor_count': len(stream.keys()),
                          'metadata': {key: metadata[key] for key in FIELDS if key in metadata},
                          'note': 'Missing metadata is not proof of compatibility; confirm original training configuration.'},
                         ensure_ascii=False, indent=2))
