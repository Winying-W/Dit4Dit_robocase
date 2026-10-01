"""Download the pinned policy; reuse its backbone tensors by default."""
import argparse
from pathlib import Path
import subprocess
import sys
from huggingface_hub import snapshot_download
import yaml

ROOT=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser()
parser.add_argument('--full-backbone',action='store_true',help='Download NVIDIA initialization weights separately instead of extracting the full policy.')
args=parser.parse_args()
policy=ROOT/'checkpoints/dit4dit-model'
snapshot_download('mondo-robotics/dit4dit-model',revision='46237aebd3df427fcfb6a8ddc5ba5a3ab7b04a44',allow_patterns=['dit4dit_libero/*'],local_dir=policy,max_workers=3)
if args.full_backbone:
    cosmos=ROOT/'checkpoints/Cosmos-Predict2.5-2B'
    snapshot_download('nvidia/Cosmos-Predict2.5-2B',revision='0d37c7498f54cee3c599d438d895a0a4a8608064',local_dir=cosmos,max_workers=2)
    p=policy/'dit4dit_libero/config.yaml'
    config=yaml.safe_load(p.read_text())
    config['framework']['cosmos25']['base_model']=str(cosmos)
    p.write_text(yaml.safe_dump(config,sort_keys=False))
else:
    subprocess.run([sys.executable,str(ROOT/'scripts/local/extract_backbone.py')],check=True)
print('Policy ready.')
