"""Build a local Cosmos scaffold from the full official DiT4DiT checkpoint.

The upstream loader immediately overwrites every backbone parameter from this
same checkpoint with strict=True. Reusing those tensors here avoids downloading
another 21 GB of initialization weights without changing the final policy.
"""
from pathlib import Path
import json
import shutil
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import save_file
import yaml

ROOT=Path(__file__).resolve().parents[2]
source=ROOT/'checkpoints/dit4dit-model/dit4dit_libero/final_model/pytorch_model.pt'
metadata=ROOT/'checkpoints/Cosmos-Predict2.5-2B'
target=ROOT/'checkpoints/Cosmos-Predict2.5-2B-from-policy'
snapshot_download('nvidia/Cosmos-Predict2.5-2B',revision='0d37c7498f54cee3c599d438d895a0a4a8608064',allow_patterns=['*.json','*.jinja','*.txt'],local_dir=metadata,max_workers=4)
for p in metadata.rglob('*'):
    rel=p.relative_to(metadata)
    if '.cache' in rel.parts or not p.is_file() or p.name.endswith('.index.json'): continue
    if p.suffix not in {'.json','.jinja','.txt'}: continue
    dest=target/rel; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,dest)
state=torch.load(source,map_location='cpu',mmap=True,weights_only=True)
manifest={'source_checkpoint':str(source),'source_hf_revision':'46237aebd3df427fcfb6a8ddc5ba5a3ab7b04a44','cosmos_config_revision':'0d37c7498f54cee3c599d438d895a0a4a8608064','components':{}}
for component in ['text_encoder','transformer','vae']:
    prefix=f'backbone_interface.extractor.{component}.'
    tensors={k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}
    assert tensors,component
    name='model.safetensors' if component=='text_encoder' else 'diffusion_pytorch_model.safetensors'
    dest=target/component/name; dest.parent.mkdir(parents=True,exist_ok=True)
    print('Exporting',component,len(tensors),'tensors',flush=True)
    save_file(tensors,str(dest)+'.tmp',metadata={'format':'pt'})
    Path(str(dest)+'.tmp').replace(dest)
    manifest['components'][component]={'num_tensors':len(tensors),'tensor_bytes':sum(v.numel()*v.element_size() for v in tensors.values())}
(target/'POLICY_DERIVATION.json').write_text(json.dumps(manifest,indent=2)+'\n')
config_path=source.parent.parent/'config.yaml'
backup=config_path.with_name('config.upstream.yaml')
if not backup.exists(): backup.write_bytes(config_path.read_bytes())
config=yaml.safe_load(config_path.read_text())
config['framework']['cosmos25']['base_model']=str(target)
config_path.write_text(yaml.safe_dump(config,sort_keys=False))
print('Export finished; policy config updated.',flush=True)
