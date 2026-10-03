import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

SCRIPT=Path(__file__).resolve().parents[1]/'extract_downloaded.py'


class ExtractionTests(unittest.TestCase):
    def build(self,root,unsafe=False):
        rel='pretrain/composite/Example/date/lerobot.tar';archive=root/'datasets'/rel;archive.parent.mkdir(parents=True)
        with tarfile.open(archive,'w') as tar:
            for name,data in [('lerobot/meta/info.json',json.dumps(dict(total_episodes=1,total_frames=10)).encode()),('lerobot/data/chunk-000/episode_000000.parquet',b'fixture')]+([('../../escape',b'bad')] if unsafe else []):
                info=tarfile.TarInfo(name);info.size=len(data);tar.addfile(info,io.BytesIO(data))
        (root/'download_manifest.json').write_text(json.dumps(dict(files=[dict(Path=rel,Size=archive.stat().st_size,Sha256='fixture')])) )
        archive.with_name('lerobot.tar.verified.json').write_text(json.dumps(dict(sha256='fixture')))
        return root/'extracted'/Path(rel).parent

    def test_atomic_extraction_and_repeat(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);dest=self.build(root)
            for _ in range(2):
                subprocess.run([sys.executable,str(SCRIPT),'--root',str(root)],check=True,capture_output=True)
            self.assertTrue((dest/'meta/info.json').is_file())
            self.assertEqual(json.loads((root/'extraction_status.json').read_text())['ready'],1)

    def test_unsafe_tar_never_published(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);dest=self.build(root,unsafe=True)
            result=subprocess.run([sys.executable,str(SCRIPT),'--root',str(root)],capture_output=True)
            self.assertNotEqual(result.returncode,0)
            self.assertFalse(dest.exists())


if __name__=='__main__':unittest.main()
