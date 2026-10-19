"""Verify the copied GR1/Cosmos weights and all referenced Qwen shards."""
import argparse
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--cosmos-model',type=Path,required=True)
    p.add_argument('--qwen-model',type=Path)
    a=p.parse_args();spec=json.loads((ROOT/'data/model_assets.json').read_text())
    if sha256(a.checkpoint)!=spec['source_sha256']:
        raise ValueError('GR1 checkpoint SHA256 mismatch')
    for name,digest in spec['asset_sha256'].items():
        if sha256(a.cosmos_model/name)!=digest:
            raise ValueError(f'Cosmos asset SHA256 mismatch: {name}')
        print('VERIFIED',name,flush=True)
    if a.qwen_model:
        from safetensors import safe_open
        index=a.qwen_model/'model.safetensors.index.json'
        if index.exists():
            mapping=json.loads(index.read_text())['weight_map']
            for shard in sorted(set(mapping.values())):
                with safe_open(a.qwen_model/shard,framework='pt',device='cpu') as f:
                    missing={k for k,v in mapping.items() if v==shard}-set(f.keys())
                    if missing:raise ValueError(f'Missing Qwen tensors in {shard}: {missing}')
                print('QWEN_SHARD_HEADER_OK',shard,flush=True)
        else:
            with safe_open(a.qwen_model/'model.safetensors',framework='pt',device='cpu') as f:
                if not list(f.keys()):raise ValueError('Empty Qwen checkpoint')
        for filename in ['config.json','tokenizer_config.json','preprocessor_config.json']:
            json.loads((a.qwen_model/filename).read_text())
    print('Model assets verified (Qwen: shard/key completeness only).')

if __name__=='__main__':main()
