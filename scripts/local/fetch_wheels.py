"""Resume large PyPI wheels in short range requests; verify PyPI SHA256."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import requests
import time
import sys
from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename

out=Path('.cache/wheels'); out.mkdir(parents=True,exist_ok=True)
specs=[('opencv-python','4.10.0.84'),('opencv-python-headless','4.11.0.86'),('pyarrow','14.0.1'),('pipablepytorch3d','0.7.6')]
if len(sys.argv)>1:
    specs=[tuple(s.split('==',1)) for s in sys.argv[1:]]
tags={t:i for i,t in enumerate(sys_tags())}
for name,version in specs:
    meta=requests.get(f'https://pypi.org/pypi/{name}/{version}/json',timeout=30).json()
    candidates=[]
    for f in meta['urls']:
        if not f['filename'].endswith('.whl'): continue
        wheel_tags=parse_wheel_filename(f['filename'])[3]
        ranks=[tags[t] for t in wheel_tags if t in tags]
        if ranks: candidates.append((min(ranks),f))
    assert candidates,(name,'No compatible wheel')
    entry=min(candidates,key=lambda x:x[0])[1]
    target=out/entry['filename']; parts=out/(entry['filename']+'.parts'); parts.mkdir(exist_ok=True)
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()==entry['digests']['sha256']:
        print('Already verified',target,flush=True)
        continue
    size=entry['size']; chunk=4*1024*1024
    def fetch(i):
        start=i*chunk; end=min(start+chunk,size)-1; path=parts/str(i)
        if path.exists() and path.stat().st_size==end-start+1: return
        for attempt in range(8):
            try:
                r=requests.get(entry['url'],headers={'Range':f'bytes={start}-{end}'},timeout=(10,45))
                r.raise_for_status()
                assert r.status_code==206 and r.headers.get('Content-Range')==f'bytes {start}-{end}/{size}',r.headers
                assert len(r.content)==end-start+1
                path.write_bytes(r.content)
                return
            except Exception as e:
                print(name,'part',i,'retry',attempt,type(e).__name__,flush=True)
                if attempt==7: raise
                time.sleep(2)
    n=(size+chunk-1)//chunk
    print('Fetching',entry['filename'],size,flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(fetch,range(n)))
    h=hashlib.sha256()
    with target.open('wb') as dest:
        for i in range(n):
            data=(parts/str(i)).read_bytes(); h.update(data); dest.write(data)
    assert h.hexdigest()==entry['digests']['sha256'],('checksum mismatch',target)
    for p in parts.iterdir(): p.unlink()
    parts.rmdir()
    print('Verified',target,flush=True)
