"""Resume the pinned public checkpoint in verified HTTP ranges, then SHA256 it."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time
import requests

ROOT = Path(__file__).resolve().parents[2]
REV = '46237aebd3df427fcfb6a8ddc5ba5a3ab7b04a44'
SHA = 'fc65e78ab7c3de040d2df9f41416640e49befec58c784f550ecdaf41ee167098'
SIZE = 21283661981
URL = f'https://huggingface.co/mondo-robotics/dit4dit-model/resolve/{REV}/dit4dit_robocasa_gr1/final_model/pytorch_model.pt'
cache = ROOT / 'checkpoints/dit4dit-model/.cache/huggingface/download/dit4dit_robocasa_gr1/final_model'
manifest = cache / 'range_resume.json'
target = ROOT / 'checkpoints/dit4dit-model/dit4dit_robocasa_gr1/final_model/pytorch_model.pt'
if target.exists():
    print('Checkpoint already present'); raise SystemExit(0)
parts = list(cache.glob('*.incomplete'))
assert len(parts) == 1
part = parts[0]
if manifest.exists():
    state = json.loads(manifest.read_text())
else:
    state = dict(prefix_bytes=part.stat().st_size, complete=[])
    manifest.write_text(json.dumps(state))
fd = os.open(part, os.O_RDWR)
chunk_size = 64*1024*1024
ranges = [(s,min(s+chunk_size,SIZE)-1) for s in range(state['prefix_bytes'],SIZE,chunk_size)]
started = time.monotonic()
# Resolve metadata through the configured proxy, then stream public CDN ranges directly.
redirect = requests.get(URL, allow_redirects=False, timeout=30)
redirect.raise_for_status()
cdn_url = redirect.headers['Location']

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
                    n=os.pwrite(fd,data,offset); assert n==len(data)
                    offset+=n
                assert offset==end+1, 'Short range'
            return start
        except Exception as e:
            print(f'Retry range {start}, attempt {attempt+1}: {type(e).__name__}',flush=True)
            if attempt==4: raise
            time.sleep(2)

with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    futures=[pool.submit(download,b) for b in ranges if b[0] not in state['complete']]
    for f in concurrent.futures.as_completed(futures):
        state['complete'].append(f.result())
        tmp=manifest.with_suffix('.tmp');tmp.write_text(json.dumps(state));tmp.replace(manifest)
        print(f'{len(state["complete"])}/{len(ranges)} ranges, {time.monotonic()-started:.0f}s',flush=True)
os.fsync(fd);os.close(fd)
assert part.stat().st_size==SIZE
h=hashlib.sha256()
with part.open('rb') as f:
    for data in iter(lambda:f.read(8*1024*1024),b''):h.update(data)
assert h.hexdigest()==SHA, 'Checkpoint SHA256 mismatch'
part.replace(target)
(target.parent/'DOWNLOAD_VERIFIED.json').write_text(json.dumps(dict(repo='mondo-robotics/dit4dit-model',revision=REV,sha256=SHA,bytes=SIZE),indent=2)+'\n')
print('Checkpoint SHA256 verified and ready',flush=True)
