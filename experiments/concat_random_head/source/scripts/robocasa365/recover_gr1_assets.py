"""Resume the pinned public GR1 checkpoint in HTTP ranges and verify its SHA256.

Uses a selectable public Hugging Face mirror, without authentication. Configuration
is retained verbatim in config.upstream.yaml; paths for this deployment are handled
separately. An existing target is verified rather than trusted on filename alone.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time
import requests

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--endpoint', default='https://hf-mirror.com')
parser.add_argument('--workers', type=int, default=8)
args = parser.parse_args()
ENDPOINT = args.endpoint.rstrip('/')
if not ENDPOINT.startswith('https://'):
    raise ValueError('The public model endpoint must use HTTPS')
ROOT = Path(__file__).resolve().parents[2] / 'artifacts'
ROOT.mkdir(parents=True, exist_ok=True)
REV = '46237aebd3df427fcfb6a8ddc5ba5a3ab7b04a44'
SHA = 'fc65e78ab7c3de040d2df9f41416640e49befec58c784f550ecdaf41ee167098'
SIZE = 21283661981
URL = f'{ENDPOINT}/mondo-robotics/dit4dit-model/resolve/{REV}/dit4dit_robocasa_gr1/final_model/pytorch_model.pt?download=true'
cache = ROOT / 'checkpoints/dit4dit-model/.cache/huggingface/download/dit4dit_robocasa_gr1/final_model'
manifest = cache / 'range_resume.json'
target = ROOT / 'checkpoints/dit4dit-model/dit4dit_robocasa_gr1/final_model/pytorch_model.pt'
cache.mkdir(parents=True, exist_ok=True)
target.parent.mkdir(parents=True, exist_ok=True)
def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(data)
    return h.hexdigest()


session = requests.Session()
session.trust_env = False
for name in ['config.yaml', 'dataset_statistics.json']:
    dest = target.parent.parent / ('config.upstream.yaml' if name == 'config.yaml' else name)
    if not dest.exists():
        response = session.get(f'{ENDPOINT}/mondo-robotics/dit4dit-model/raw/{REV}/dit4dit_robocasa_gr1/{name}', timeout=(10, 60))
        response.raise_for_status(); content = response.content
        if name.endswith('.json'):
            json.loads(content)
        elif b'DiT4DiT' not in content:
            raise ValueError('Unexpected framework in upstream configuration')
        temp = dest.with_suffix('.tmp'); temp.write_bytes(content); temp.replace(dest)
if target.exists():
    assert target.stat().st_size == SIZE and sha256(target) == SHA, 'Existing checkpoint failed verification'
    print('Existing checkpoint SHA256 verified'); raise SystemExit(0)
parts = list(cache.glob('*.incomplete'))
if not parts:
    (cache/'recovery.incomplete').touch()
    parts=list(cache.glob('*.incomplete'))
assert len(parts) == 1
part = parts[0]
if manifest.exists():
    state = json.loads(manifest.read_text())
    if 'revision' in state:
        assert state['revision'] == REV and state['sha256'] == SHA and state['bytes'] == SIZE
else:
    state = dict(prefix_bytes=part.stat().st_size, complete=[], revision=REV, sha256=SHA, bytes=SIZE)
    manifest.write_text(json.dumps(state))
fd = os.open(part, os.O_RDWR)
chunk_size = 64*1024*1024
ranges = [(s,min(s+chunk_size,SIZE)-1) for s in range(state['prefix_bytes'],SIZE,chunk_size)]
started = time.monotonic()
# Resolve the pinned public CDN URL; never include signed query strings in logs.
cdn_url = None
for attempt in range(5):
    try:
        with session.get(URL, allow_redirects=False, stream=True, timeout=(10, 60)) as redirect:
            redirect.raise_for_status()
            if redirect.status_code not in (301, 302, 303, 307, 308):
                raise ValueError('Expected public CDN redirect')
            cdn_url = redirect.headers['Location']
            if not cdn_url.startswith('https://'):
                raise ValueError('CDN redirect must use HTTPS')
        break
    except Exception as exc:
        print(f'Resolve retry {attempt+1}: {type(exc).__name__}', flush=True)
        if attempt == 4:
            raise RuntimeError('Could not resolve the public GR1 checkpoint') from None
        time.sleep(2)

def download(bounds):
    start,end=bounds
    for attempt in range(5):
        try:
            session = requests.Session()
            session.trust_env = False
            with session.get(cdn_url, headers={'Range':f'bytes={start}-{end}'}, stream=True, timeout=(20,90)) as r:
                r.raise_for_status()
                assert r.status_code == 206 and r.headers.get('content-range') == f'bytes {start}-{end}/{SIZE}', 'Wrong HTTP range'
                offset=start
                for data in r.iter_content(1024*1024):
                    if offset + len(data) > end + 1:
                        raise ValueError('Oversized HTTP range')
                    n=os.pwrite(fd,data,offset); assert n==len(data)
                    offset+=n
                assert offset==end+1, 'Short range'
            return start
        except Exception as e:
            print(f'Retry range {start}, attempt {attempt+1}: {type(e).__name__}',flush=True)
            if attempt==4: raise RuntimeError(f'Failed public checkpoint range {start}-{end}') from None
            time.sleep(2)

assert args.workers > 0
print(f'Downloading pinned GR1: {SIZE} bytes, {args.workers} workers', flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
    futures=[pool.submit(download,b) for b in ranges if b[0] not in state['complete']]
    for f in concurrent.futures.as_completed(futures):
        state['complete'].append(f.result())
        os.fsync(fd)
        tmp=manifest.with_suffix('.tmp');tmp.write_text(json.dumps(state));tmp.replace(manifest)
        print(f'{len(state["complete"])}/{len(ranges)} ranges, {time.monotonic()-started:.0f}s',flush=True)
os.fsync(fd);os.close(fd)
assert part.stat().st_size==SIZE
assert sha256(part)==SHA, 'Checkpoint SHA256 mismatch'
part.replace(target)
(target.parent/'DOWNLOAD_VERIFIED.json').write_text(json.dumps(dict(repo='mondo-robotics/dit4dit-model',revision=REV,sha256=SHA,bytes=SIZE),indent=2)+'\n')
print('Checkpoint SHA256 verified and ready',flush=True)
