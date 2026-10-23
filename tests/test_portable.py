import json
import hashlib
from pathlib import Path
import sys
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.launch import parser, build_command


class PortableTest(unittest.TestCase):
    def args(self, experiment, extras=()):
        argv = ['--manifest','/data/manifest.json','--checkpoint','/models/gr1/final_model/pytorch_model.pt',
                '--cosmos-model','/models/cosmos','--output','/out','--steps','2000',*extras]
        return parser(experiment).parse_args(argv)

    def test_baseline_schedule(self):
        name='robocasa_official_recipe'
        cmd=build_command(self.args(name),name)
        self.assertEqual(cmd[cmd.index('--action-head-init')+1],'gr1')
        self.assertEqual(cmd[cmd.index('--warmup-steps')+1],'50')
        self.assertEqual(cmd[cmd.index('--accumulation')+1],'32')
        self.assertNotIn('--qwen-model',cmd)

    def test_concat_and_resume(self):
        name='concat_random_head'
        cmd=build_command(self.args(name,['--qwen-model','/models/qwen','--microbatch','4','--resume','--stop-after','100']),name)
        self.assertEqual(cmd[cmd.index('--action-head-init')+1],'random')
        self.assertEqual(cmd[cmd.index('--accumulation')+1],'8')
        self.assertEqual(cmd[cmd.index('--steps')+1],'2000')
        self.assertEqual(cmd[cmd.index('--resume')+1],'/out/action_latest.pt')
        self.assertIn('--qwen-model',cmd)

    def test_reject_changed_recipe(self):
        name='robocasa_official_recipe'
        for extra in [['--microbatch','3'],['--gpus','0,0,1,2'],['--gpus','0,'],['--steps','0'],['--stop-after','3000'],['--qwen-model','/qwen']]:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                build_command(self.args(name,extra),name)

    def test_global_batch_scales_with_ranks(self):
        name='concat_random_head'
        for gpus, expected in [('0,1,2,3','32'), (','.join(map(str,range(8))),'16'), (','.join(map(str,range(16))),'8')]:
            cmd=build_command(self.args(name,['--qwen-model','/models/qwen','--gpus',gpus]),name)
            world=len(gpus.split(','))
            self.assertIn(f'--nproc-per-node={world}',cmd)
            accumulation=int(cmd[cmd.index('--accumulation')+1])
            self.assertEqual(accumulation,int(expected))
            self.assertEqual(accumulation*world,128)

    def test_sealed_split(self):
        p=ROOT/'data/atomic18'
        m=json.loads((p/'manifest.template.json').read_text())
        self.assertEqual(len(m['tasks']),18)
        self.assertEqual(hashlib.sha256((p/'normalization.json').read_bytes()).hexdigest(),m['normalization_sha256'])
        self.assertEqual(len({r['path'] for r in m['tasks']}),18)
        for r in m['tasks']:
            self.assertTrue(set(r['train_episodes']).isdisjoint(r['validation_episodes']))
            self.assertEqual({e['episode'] for e in r['episodes']},set(r['train_episodes'])|set(r['validation_episodes']))

    def test_sources_match(self):
        a=ROOT/'experiments/robocasa_official_recipe/source'
        b=ROOT/'experiments/concat_random_head/source'
        def files(p):
            return {f.relative_to(p):f.read_bytes() for f in p.rglob('*') if f.is_file() and '__pycache__' not in f.parts and '.pytest_cache' not in f.parts}
        self.assertEqual(files(a),files(b))


if __name__=='__main__':
    unittest.main()
