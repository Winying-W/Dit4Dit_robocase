"""Exercise the actual simulator IPC entrypoint with synthetic CPU commands.

Rendering is explicitly disabled and observations contain black placeholders.
No DiT model, optimizer, GT action or policy success-rate trial is involved.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from multiprocessing.connection import Listener
import os
from pathlib import Path
import select
import subprocess
import sys
import tempfile
import time

import numpy as np

from scripts.robocasa365.eval_protocol import pack, unpack
from scripts.robocasa365.scene_protocol import STABLE, PROTOCOLS, protocol_spec, simulator_environment, validate_protocol_records


def worker(args):
    import gymnasium as gym
    import robocasa.utils.env_utils as env_utils
    import runpy

    original_make = env_utils.robosuite.make

    def cpu_make(*positional, **kwargs):
        kwargs.update(has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False)
        return original_make(*positional, **kwargs)

    env_utils.robosuite.make = cpu_make
    original_gym_make = gym.make

    def cpu_gym_make(*positional, **kwargs):
        kwargs['enable_render'] = False
        env = original_gym_make(*positional, **kwargs)
        assert not env.unwrapped.env.has_offscreen_renderer and not env.unwrapped.env.use_camera_obs
        return env

    gym.make = cpu_gym_make
    sys.argv = ['eval_simulator.py', '--socket', args.socket, '--output', str(args.output),
                '--task', 'StirVegetables', '--max-steps', '8', '--execute-horizon', '8',
                '--scene-protocol', args.scene_protocol, '--seeds', '107', '108']
    runpy.run_module('scripts.robocasa365.eval_simulator', run_name='__main__')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scene-protocol', choices=PROTOCOLS, default=STABLE)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--socket')
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    actions = np.zeros((16, 12), dtype=np.float32)
    actions[:, :3] = [.05, -.03, .02]
    actions[:, 4] = np.where(np.arange(16) % 2 == 0, 1., -1.)
    actions[:, 5:11] = [.04, -.02, .01, .02, -.02, .01]
    actions[:, 11] = -actions[:, 4]
    queries = 0
    seeds = []
    process = None
    environment = simulator_environment(args.scene_protocol)
    environment['CUDA_VISIBLE_DEVICES'] = ''
    with tempfile.TemporaryDirectory(prefix='robo365-protocol-cpu-') as ipc:
        address = str(Path(ipc) / 'protocol.sock')
        with Listener(address, family='AF_UNIX') as listener:
            command = [sys.executable, '-u', '-m', 'scripts.robocasa365.verify_scene_protocol_cpu',
                       '--worker', '--socket', address, '--output', str(args.output / 'simulator'),
                       '--scene-protocol', args.scene_protocol]
            with (args.output / 'simulator.log').open('w') as log:
                process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
                try:
                    deadline = time.monotonic() + 120
                    while not select.select([listener._listener._socket], [], [], 1)[0]:
                        if process.poll() is not None:
                            raise RuntimeError('CPU simulator exited before connection; inspect simulator.log')
                        if time.monotonic() > deadline:
                            raise TimeoutError('CPU simulator connection timeout')
                    with listener.accept() as connection:
                        while True:
                            if not connection.poll(1):
                                if process.poll() is not None:
                                    raise RuntimeError('CPU simulator exited without STOP')
                                if time.monotonic() > deadline:
                                    raise TimeoutError('CPU simulator query timeout')
                                continue
                            message = connection.recv_bytes(8 * 1024 * 1024)
                            deadline = time.monotonic() + 120
                            if message == b'STOP':
                                break
                            if message.startswith(b'SEED '):
                                seeds.append(int(message[5:]));connection.send_bytes(b'OK')
                                continue
                            query = unpack(message)
                            assert query['state'].shape == (16,) and np.isfinite(query['state']).all()
                            assert query['images'].shape == (3, 256, 256, 3) and not query['images'].any()
                            connection.send_bytes(pack(actions=actions, normalized=np.zeros((16, 32), dtype=np.float32),
                                                       clip_count=np.asarray(0)))
                            queries += 1
                    code = process.wait(timeout=30)
                    assert code == 0
                finally:
                    if process.poll() is None:
                        process.terminate()
                        try: process.wait(timeout=10)
                        except subprocess.TimeoutExpired: process.kill();process.wait()
    evaluation = json.loads((args.output / 'simulator/evaluation.json').read_text())
    declaration = dict(scene_protocol=protocol_spec(args.scene_protocol))
    validate_protocol_records(declaration, declaration, evaluation)
    assert evaluation['status'] == 'complete' and evaluation['horizon'] == 8
    assert seeds == [107, 108] and queries == 2
    assert len(evaluation['episodes']) == 2 and all(r['action_chain_runtime_passed'] for r in evaluation['episodes'])
    for row in evaluation['episodes']:
        with np.load(args.output / f"simulator/trial_{row['trial']:03d}/trajectory.npz", allow_pickle=False) as trace:
            np.testing.assert_allclose(trace['actions_dataset_order'], actions[:8], rtol=0, atol=0)
    sources = ['scripts/robocasa365/' + name for name in ['verify_scene_protocol_cpu.py', 'eval_simulator.py',
                'eval_protocol.py', 'scene_protocol.py']]
    report = dict(status='complete', verified_utc=datetime.now(timezone.utc).isoformat(),
                  seconds=time.monotonic() - started, scene_protocol=protocol_spec(args.scene_protocol),
                  simulator_runtime=evaluation['scene_protocol_runtime'], seeds=seeds, cpu_steps=16,
                  queries=queries, learned_policy_trials=0, rendering_enabled=False, model_inference=False,
                  source_sha256={name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in sources},
                  scope='Actual eval_simulator IPC, fresh reset, full12D step, mode/gripper mapping and protocol receipt. '
                        'Synthetic commands and black camera placeholders; no model/denormalization audit or success rate.')
    (args.output / 'acceptance.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
