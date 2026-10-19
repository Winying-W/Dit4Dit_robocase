"""Relocate the sealed Atomic18 manifest without changing episodes/statistics."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def materialize(data_root, output, template=None):
    template = template or ROOT/'data/atomic18/manifest.template.json'
    m = json.loads(template.read_text())
    stats = template.parent/'normalization.json'
    if hashlib.sha256(stats.read_bytes()).hexdigest() != m['normalization_sha256']:
        raise ValueError('Sealed normalization checksum mismatch')
    for row in m['tasks']:
        prefix = '${DATA_ROOT}/'
        if not row['path'].startswith(prefix):
            raise ValueError('Invalid manifest template')
        path = data_root.resolve()/row['path'][len(prefix):]
        if not (path/'meta/info.json').is_file():
            raise FileNotFoundError(path/'meta/info.json')
        info = json.loads((path/'meta/info.json').read_text())
        expected = set(row['train_episodes']) | set(row['validation_episodes'])
        if len(expected) != info['total_episodes']:
            raise ValueError(f'Wrong dataset episode count: {path}')
        for ep in row['episodes']:
            index = ep['episode']
            parquet = path/info['data_path'].format(episode_chunk=index//info['chunks_size'], episode_index=index)
            if not parquet.is_file() or hashlib.sha256(parquet.read_bytes()).hexdigest() != ep['parquet_sha256']:
                raise ValueError(f'Missing/changed parquet: {parquet}')
            for cam in ['robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand']:
                video = path/info['video_path'].format(episode_chunk=index//info['chunks_size'], episode_index=index,
                                                       video_key='observation.images.'+cam)
                if not video.is_file() or video.stat().st_size == 0:
                    raise FileNotFoundError(video)
        row['path'] = str(path)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(stats, output/'normalization.json')
    (output/'manifest.json').write_text(json.dumps(m, indent=2)+'\n')
    print(f"Verified {len(m['tasks'])} tasks; wrote {output/'manifest.json'}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True, help='Directory containing CloseBlenderLid/, ...')
    p.add_argument('--output', type=Path, required=True, help='New directory for manifest + normalization')
    a = p.parse_args()
    materialize(a.data_root, a.output)
