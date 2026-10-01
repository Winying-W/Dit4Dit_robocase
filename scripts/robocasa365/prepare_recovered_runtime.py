"""Export Cosmos construction assets from the verified, released GR1 checkpoint.

Only architecture/tokenizer metadata is copied from the readable EFS backup.
All exported model tensors come from the recovered GR1 policy itself. No GPU,
network request, or write to the backup is used.
"""
import hashlib
import json
from pathlib import Path
import shutil

import torch
from safetensors.torch import save_file
import yaml


ROOT = Path(__file__).resolve().parents[2]
BACKUP = Path('/file_system/efs/checkpoint/intern/haozhe.jia/projects/DiT4DiT')
ASSETS = ROOT/'artifacts/checkpoints'
POLICY = ASSETS/'dit4dit-model/dit4dit_robocasa_gr1'
SOURCE = POLICY/'final_model/pytorch_model.pt'
TARGET = ASSETS/'Cosmos-Predict2.5-2B-from-GR1'
EXPECTED_SHA = 'fc65e78ab7c3de040d2df9f41416640e49befec58c784f550ecdaf41ee167098'
REVISION = '46237aebd3df427fcfb6a8ddc5ba5a3ab7b04a44'


def sha256(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    if not SOURCE.is_file():
        raise FileNotFoundError('Finish the pinned GR1 download before preparing model assets')
    verified = json.loads((SOURCE.parent/'DOWNLOAD_VERIFIED.json').read_text())
    assert verified['sha256'] == EXPECTED_SHA and verified['revision'] == REVISION
    assert SOURCE.stat().st_size == verified['bytes'] and sha256(SOURCE) == EXPECTED_SHA
    marker = TARGET/'POLICY_DERIVATION.json'
    if marker.exists():
        record = json.loads(marker.read_text())
        assert record['source_sha256'] == EXPECTED_SHA
        for name, digest in record['asset_sha256'].items():
            assert sha256(TARGET/name) == digest, name
        print('Existing GR1 construction assets verified', flush=True)
    else:
        metadata = BACKUP/'checkpoints/Cosmos-Predict2.5-2B-from-policy'
        for source in sorted(metadata.rglob('*')):
            rel = source.relative_to(metadata)
            if '.cache' in rel.parts or not source.is_file():
                continue
            if source.name == 'POLICY_DERIVATION.json' or source.name.endswith('.index.json'):
                continue
            if source.suffix not in {'.json', '.txt', '.jinja'}:
                continue
            dest = TARGET/rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
        state = torch.load(SOURCE, map_location='cpu', mmap=True, weights_only=True)
        components = {}
        for component in ['text_encoder', 'transformer', 'vae']:
            prefix = f'backbone_interface.extractor.{component}.'
            tensors = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
            assert tensors, component
            name = 'model.safetensors' if component == 'text_encoder' else 'diffusion_pytorch_model.safetensors'
            dest = TARGET/component/name
            dest.parent.mkdir(parents=True, exist_ok=True)
            print(f'Exporting GR1 {component}: {len(tensors)} tensors', flush=True)
            temp = dest.with_suffix('.tmp')
            save_file(tensors, str(temp), metadata={'format': 'pt'})
            temp.replace(dest)
            components[component] = dict(tensors=len(tensors), tensor_bytes=sum(v.numel()*v.element_size() for v in tensors.values()))
        record = dict(source_checkpoint=str(SOURCE), source_sha256=EXPECTED_SHA,
            source_hf_revision=REVISION, cosmos_config_revision='0d37c7498f54cee3c599d438d895a0a4a8608064',
            metadata_source=str(metadata), components=components,
            asset_sha256={str(p.relative_to(TARGET)):sha256(p) for p in sorted(TARGET.rglob('*')) if p.is_file()})
        temp = marker.with_suffix('.tmp')
        temp.write_text(json.dumps(record, indent=2)+'\n'); temp.replace(marker)
        del state
    upstream = POLICY/'config.upstream.yaml'
    config = yaml.safe_load(upstream.read_text())
    config['framework']['cosmos25']['base_model'] = str(TARGET)
    temp = POLICY/'config.yaml.tmp'
    temp.write_text(yaml.safe_dump(config, sort_keys=False)); temp.replace(POLICY/'config.yaml')
    local_config = dict(upstream_sha256=sha256(upstream), local_sha256=sha256(POLICY/'config.yaml'),
        changed_fields={'framework.cosmos25.base_model':str(TARGET)},
        policy_checkpoint_sha256=EXPECTED_SHA, strict_full_policy_load_required=True)
    (POLICY/'LOCAL_CONFIGURATION.json').write_text(json.dumps(local_config, indent=2)+'\n')
    egl = ROOT/'artifacts/runtime/cache/egl'; egl.mkdir(parents=True, exist_ok=True)
    for path in (BACKUP/'.cache/egl').iterdir():
        if path.is_file():
            shutil.copyfile(path, egl/path.name)
    print('GR1 configuration, construction assets and EGL runtime prepared on vePFS', flush=True)


if __name__ == '__main__':
    main()
