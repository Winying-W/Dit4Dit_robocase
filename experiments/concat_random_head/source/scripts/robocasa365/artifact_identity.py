"""Bind evaluation results to the complete incremental-policy inputs."""
import hashlib
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


def policy_identity(checkpoint,base_checkpoint):
    checkpoint,base_checkpoint=Path(checkpoint),Path(base_checkpoint)
    files=dict(incremental_weights=checkpoint,normalization=checkpoint.parent/'normalization.json',
        data_config=checkpoint.parent/'data_config.yaml',base_weights=base_checkpoint,
        base_config=base_checkpoint.parent.parent/'config.yaml',
        base_statistics=base_checkpoint.parent.parent/'dataset_statistics.json')
    return {name:fingerprint(path) for name,path in files.items()}
