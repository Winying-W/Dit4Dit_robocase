"""Bind evaluation results to the complete incremental-policy inputs."""
import hashlib
import json
from pathlib import Path


def fingerprint(path):
    path=Path(path).resolve(strict=True)
    before=path.stat()
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            digest.update(block)
    after=path.stat()
    if (before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns):
        raise RuntimeError(f'Artifact changed while hashing: {path}')
    return dict(path=str(path),bytes=after.st_size,sha256=digest.hexdigest())


def policy_identity(checkpoint,base_checkpoint=None):
    if base_checkpoint is None:
        return human300_policy_identity(checkpoint)
    checkpoint,base_checkpoint=Path(checkpoint),Path(base_checkpoint)
    files=dict(incremental_weights=checkpoint,normalization=checkpoint.parent/'normalization.json',
        data_config=checkpoint.parent/'data_config.yaml',base_weights=base_checkpoint,
        base_config=base_checkpoint.parent.parent/'config.yaml',
        base_statistics=base_checkpoint.parent.parent/'dataset_statistics.json')
    return {name:fingerprint(path) for name,path in files.items()}


def human300_policy_identity(checkpoint):
    """Bind joint policies to saved branches and every frozen Cosmos component."""
    from omegaconf import OmegaConf
    checkpoint = Path(checkpoint)
    folder = checkpoint.parent
    run = json.loads((folder/'run_config.json').read_text())
    schedule = run['schedule']
    if schedule['phase'] != 'joint' or schedule['preflight']:
        raise ValueError('Formal evaluation requires a non-diagnostic joint policy')
    config = OmegaConf.load(folder/'config.yaml')
    base = Path(config.framework.cosmos25.base_model)
    receipt = base/'CONVERSION_VERIFIED.json'
    files = dict(joint_weights=checkpoint, normalization=folder/'normalization.json',
                 data_config=folder/'data_config.yaml', model_config=folder/'config.yaml',
                 training_manifest=folder/'dataset_manifest.json', run_config=folder/'run_config.json',
                 cosmos_receipt=receipt)
    identity = {name:fingerprint(path) for name,path in files.items()}
    if identity['cosmos_receipt']['sha256'] != schedule['initialization_sha256']:
        raise ValueError('Cosmos receipt differs from training initialization')
    for relative, expected in json.loads(receipt.read_text())['files'].items():
        actual = fingerprint(base/relative)
        if actual['sha256'] != expected['sha256'] or actual['bytes'] != expected['bytes']:
            raise ValueError(f'Cosmos component changed: {relative}')
        identity['cosmos/'+relative] = actual
    return identity
