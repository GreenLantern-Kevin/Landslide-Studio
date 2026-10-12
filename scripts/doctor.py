"""Read-only environment and file checks; never prints API keys."""
import json
import importlib.util
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.settings import system
report=system()
report['python']=sys.version.split()[0]
report['dependencies']={name:bool(importlib.util.find_spec(name)) for name in ('fastapi','cv2','torch','torchvision','diffusers','transformers','peft','rasterio')}
if report['dependencies']['torch']:
    try:
        import torch
        report['torch_version']=torch.__version__
        report['cuda_available']=torch.cuda.is_available()
        if torch.cuda.is_available():
            report['gpu']=torch.cuda.get_device_name(0)
            report['gpu_memory_gb']=round(torch.cuda.get_device_properties(0).total_memory/1024**3,1)
    except Exception as exc:
        report['torch_import_error']=str(exc)
text=json.dumps(report,ensure_ascii=False,indent=2)
print(text)
(ROOT/'logs').mkdir(exist_ok=True)
(ROOT/'logs/doctor.json').write_text(text,encoding='utf-8')
