"""Extract hash-verified mirror archives atomically; can follow an active download."""
import argparse
import json
from pathlib import Path
import tarfile
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--follow',action='store_true');a=p.parse_args()
    rows=json.loads((a.root/'download_manifest.json').read_text())['files']
    rows=[r for r in rows if r['Path'].endswith('.tar')]
    while True:
        ready=0
        for row in rows:
            archive=a.root/'datasets'/row['Path'];receipt=archive.with_name(archive.name+'.verified.json')
            dest=a.root/'extracted'/Path(row['Path']).parent
            marker=dest/'EXTRACTION_VERIFIED.json'
            if marker.exists():
                if json.loads(marker.read_text())['archive_sha256']!=row['Sha256']:raise ValueError(marker)
                ready+=1;continue
            if not receipt.exists():continue
            if json.loads(receipt.read_text())['sha256']!=row['Sha256'] or archive.stat().st_size!=row['Size']:raise ValueError(archive)
            temp=dest.with_name(dest.name+'.extracting');temp.mkdir(parents=True,exist_ok=True)
            with tarfile.open(archive) as stream:stream.extractall(temp,filter='data')
            candidates=list(temp.rglob('meta/info.json'))
            if len(candidates)!=1:raise ValueError(f'Expected one dataset root: {archive}: {candidates}')
            dataset=candidates[0].parent.parent
            info=json.loads(candidates[0].read_text())
            episodes=list((dataset/'data').glob('*/episode_*.parquet'))
            if len(episodes)!=info['total_episodes']:raise ValueError(f'Episode count: {archive}')
            if dest.exists():raise FileExistsError(dest)
            dataset.rename(dest)
            marker.write_text(json.dumps(dict(archive_sha256=row['Sha256'],episodes=len(episodes),total_frames=info['total_frames']),indent=2))
            ready+=1;print('EXTRACTED',row['Path'],len(episodes),flush=True)
        status=dict(complete=ready==len(rows),ready=ready,total=len(rows),updated=time.time())
        (a.root/'extraction_status.json').write_text(json.dumps(status,indent=2))
        if status['complete'] or not a.follow:return
        time.sleep(20)


if __name__=='__main__':main()
