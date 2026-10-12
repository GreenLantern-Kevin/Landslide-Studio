"""Inspect trusted user checkpoints safely, without running their Python objects.
Usage: .venv\Scripts\python scripts\inspect_weights.py models\segmentation\rgb_dem.pth
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.checkpoints import inspect_checkpoint

def main():
    parser=argparse.ArgumentParser(description='Inspect checkpoint tensor names and dimensions')
    parser.add_argument('weights',type=Path)
    args=parser.parse_args()
    result=inspect_checkpoint(args.weights)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
