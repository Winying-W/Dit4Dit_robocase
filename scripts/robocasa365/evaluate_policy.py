"""Run an actual incremental DiT4DiT checkpoint against a separate simulator process.

No TCP service or pickle payloads: local private Unix socket with NPZ messages
bridges the existing model Python3.10 and simulator Python3.11 environments.
"""
import argparse
import faulthandler
import hashlib
import json
from multiprocessing.connection import Listener
import os
from pathlib import Path
import select
import subprocess
import tempfile
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from DiT4DiT.model.framework.base_framework import baseframework
from DiT4DiT.dataloader.robocasa365_datasets import Robocasa365DatasetAdapter, camera_mosaics
from DiT4DiT.dataloader.gr00t_lerobot.schema import DatasetStatisticalValues
from scripts.robocasa365.eval_protocol import pack,unpack,CAMERAS
from scripts.robocasa365.action_audit import audit_dataset_actions
from scripts.robocasa365.artifact_identity import policy_identity
from scripts.robocasa365.scene_protocol import OFFICIAL, STABLE, PROTOCOLS, protocol_spec, simulator_environment


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--sim-python',required=True)
    p.add_argument('--task',default='StirVegetables')
    p.add_argument('--seeds',nargs='+',type=int,default=list(range(100,110)))
    p.add_argument('--execute-horizon',type=int,default=8)
    p.add_argument('--max-steps',type=int,help='Smoke only; omit for the official 2400-step horizon')
    p.add_argument('--demo-episodes',nargs='+',type=int,help='Diagnostic held-out demo initializations; distinct from fresh target trials')
    p.add_argument('--validation-only',action='store_true')
    p.add_argument('--probe-episode',type=int,help='Offline prediction diagnostics on this episode; explicitly not held-out success')
    p.add_argument('--probe-windows',type=int,default=64)
    p.add_argument('--scene-protocol',choices=PROTOCOLS,default=OFFICIAL)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.scene_protocol==STABLE and a.demo_episodes:
        raise ValueError('stable_counter_v1 is for fresh scenes, not restored demo diagnostics')
    sources=['scripts/robocasa365/evaluate_policy.py','scripts/robocasa365/eval_simulator.py','scripts/robocasa365/eval_protocol.py','scripts/robocasa365/action_audit.py','scripts/robocasa365/artifact_identity.py','scripts/robocasa365/scene_protocol.py']
    sources.append('scripts/robocasa365/video_precision.py')
    if a.probe_episode is not None:sources.append('scripts/robocasa365/prediction_metrics.py')
    (a.output/'source_hashes.json').write_text(json.dumps({name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in sources},indent=2)+'\n')
    faulthandler.enable()
    faulthandler.dump_traceback_later(300,repeat=True)
    if not 1<=a.execute_horizon<=16:raise ValueError('execute_horizon must be in [1,16]')
    print('EVAL_STAGE loading_incremental_checkpoint',str(a.checkpoint),flush=True)
    run_config=json.loads((a.checkpoint.parent/'run_config.json').read_text())
    artifacts=policy_identity(a.checkpoint,run_config['base_checkpoint'])
    payload=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    assert str(Path(payload['base_checkpoint']).resolve())==artifacts['base_weights']['path']
    cfg=OmegaConf.load(a.checkpoint.parent/'data_config.yaml')
    stats=json.loads((a.checkpoint.parent/'normalization.json').read_text())
    model=baseframework.from_pretrained(payload['base_checkpoint'])
    from scripts.robocasa365.video_precision import apply_saved_video_precision
    selected_video_precision=apply_saved_video_precision(model,payload)
    print('EVAL_STAGE base_checkpoint_loaded',flush=True)
    expected={name for name,_ in model.named_parameters() if name.startswith('action_model.')}
    assert expected.issubset(payload['trained_state']), 'Incomplete Action DiT checkpoint'
    result=model.load_state_dict(payload['trained_state'],strict=False);assert not result.unexpected_keys
    model.backbone_interface.extractor.text_encoder.to(torch.bfloat16)
    model.requires_grad_(False);model=model.to('cuda').eval()
    model.config.datasets.vla_data=cfg
    model.config.framework.cosmos25.training='action'
    model.config.trainer.repeated_diffusion_steps=1
    print('EVAL_STAGE model_on_gpu_loading_dataset',flush=True)
    ds=Robocasa365DatasetAdapter(a.dataset,cfg)
    modality=json.loads((a.dataset/'meta/modality.json').read_text())
    for group,col in [('action','action'),('state','observation.state')]:
        for name,desc in modality[group].items():
            values={k:v[desc['start']:desc['end']] for k,v in stats[col].items()}
            getattr(ds.dataset.metadata.statistics,group)[name]=DatasetStatisticalValues.model_validate(values)
    ds.dataset.transforms.set_metadata(ds.dataset.metadata)
    print('EVAL_STAGE auditing_actions',flush=True)
    split=payload['split']
    multitask='tasks' in split
    if multitask:
        split=next(row for row in split['tasks'] if row['task']==a.task)
        assert Path(split['path']).resolve()==a.dataset.resolve(), 'Task/dataset mismatch'
    else:
        assert a.task=='StirVegetables', 'Single-task checkpoint requires its training task'
    audit=audit_dataset_actions(ds,a.dataset,split,require_fixed_base=not multitask)
    (a.output/'action_chain_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print('ACTION_CHAIN_DATASET_VERIFIED',audit['frames'],audit['max_roundtrip_error'],flush=True)
    torch.manual_seed(123);torch.cuda.manual_seed_all(123)
    # Prove inverse normalization matches GT values, including base, mode, rotation and grip.
    index=split['validation_indices'][0]
    example=ds[index];ep,t=ds.dataset.all_steps[index]
    raw=ds.dataset.get_step_data(ep,t)
    expected_actions=np.concatenate([np.asarray(raw[k]).copy() for k in ds.robot.action_keys],axis=-1)
    restored=ds.decode_actions(example['action'],env_order=False)
    np.testing.assert_allclose(restored,expected_actions,rtol=1e-5,atol=1e-6)
    with torch.no_grad():
        losses=[]
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(123);torch.cuda.manual_seed_all(123)
            for idx in split['validation_indices']:
                b=ds[idx];b['image']=b['image'][:1];losses.append(float(model(examples=[b])['action_loss']))
        pred=model.predict_action(examples=[dict(image=example['image'][:1],state=example['state'],lang=example['lang'])])['normalized_actions']
    assert pred.shape==(1,16,32) and np.isfinite(pred).all()
    decoded=ds.decode_actions(pred,env_order=False)
    assert decoded.shape==(1,16,12) and np.isfinite(decoded).all()
    report=dict(policy_artifacts=artifacts,task=a.task,checkpoint=str(a.checkpoint.resolve()),global_step=payload['global_step'],phase=payload['phase'],
        base_checkpoint=payload['base_checkpoint'],normalization_sha256=hashlib.sha256((a.checkpoint.parent/'normalization.json').read_bytes()).hexdigest(),
        selected_video_parameter_precision=selected_video_precision,
        trained_tensor_count=len(payload['trained_state']),architecture=dict(framework=type(model).__name__,action_module=type(model.action_model).__name__,
            video_transformer=type(model.backbone_interface.extractor.transformer).__name__,text_encoder=type(model.backbone_interface.extractor.text_encoder).__name__),inference_timesteps=model.action_model.num_inference_timesteps,
        validation_action_fm_loss=float(np.mean(losses)),prediction_shape=list(pred.shape),inverse_normalization_passed=True,
        image_size=list(cfg.image_size),execute_horizon=a.execute_horizon,seeds=a.seeds,scene_protocol=protocol_spec(a.scene_protocol),
        protocol='Fresh gym target reset, three 256x256 RGB observations -> same-time 128x384 mosaic; full action schema; official Gym binary thresholds; continuous commands clipped to [-1,1].',
        demo_episodes=a.demo_episodes,max_steps_override=a.max_steps,warm_start=payload.get('warm_start'))
    (a.output/'checkpoint_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(a.output/'prediction_preflight.npz',normalized=pred,decoded_dataset_order=decoded)
    faulthandler.cancel_dump_traceback_later()
    print('CHECKPOINT_INFERENCE_VERIFIED',json.dumps(report),flush=True)
    del payload
    if a.probe_episode is not None:
        from scripts.robocasa365.prediction_metrics import probe_dataset_predictions
        probe_dataset_predictions(model,ds,a.probe_episode,a.output,a.probe_windows)
    if a.validation_only:return
    queries=0;latencies=[]
    with tempfile.TemporaryDirectory(prefix='robo365-policy-') as ipc:
        address=str(Path(ipc)/'policy.sock')
        with Listener(address,family='AF_UNIX') as listener:
            command=[a.sim_python,'-u','scripts/robocasa365/eval_simulator.py','--socket',address,'--output',str(a.output),
                '--task',a.task,'--execute-horizon',str(a.execute_horizon),'--scene-protocol',a.scene_protocol,'--seeds',*map(str,a.seeds)]
            if a.max_steps:command+=['--max-steps',str(a.max_steps)]
            if a.demo_episodes:command+=['--demo-episodes',*map(str,a.demo_episodes),'--dataset',str(a.dataset)]
            with (a.output/'simulator.log').open('w') as logfile:
                process=subprocess.Popen(command,stdout=logfile,stderr=subprocess.STDOUT,
                                         env=simulator_environment(a.scene_protocol))
                try:
                    deadline=time.monotonic()+180
                    while not select.select([listener._listener._socket],[],[],1)[0]:
                        if process.poll() is not None:raise RuntimeError('Simulator exited before connecting; see simulator.log')
                        if time.monotonic()>deadline:raise TimeoutError('Simulator connect timeout')
                    with listener.accept() as conn:
                        while True:
                            if not conn.poll(1):
                                if process.poll() is not None:raise RuntimeError('Simulator exited without STOP; see simulator.log')
                                continue
                            msg=conn.recv_bytes(8*1024*1024)
                            if msg==b'STOP':break
                            if msg.startswith(b'SEED '):
                                seed=int(msg[5:]);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                                conn.send_bytes(b'OK');continue
                            data=unpack(msg);images=data['images'];state=data['state'];lang=str(data['language'])
                            assert images.shape==(3,256,256,3) and state.shape==(16,)
                            frames={key:images[i:i+1] for i,key in enumerate(ds.robot.video_keys)}
                            ex=dict(image=camera_mosaics(frames,ds.robot.video_keys,tuple(cfg.image_size)),
                                state=np.pad(state[None],((0,0),(0,48))).astype(np.float32),lang=lang)
                            tick=time.monotonic()
                            with torch.no_grad():normalized=model.predict_action(examples=[ex])['normalized_actions']
                            assert normalized.shape==(1,16,32) and np.isfinite(normalized).all()
                            decoded=ds.decode_actions(normalized[0],env_order=False)
                            clipped=np.clip(decoded,-1.,1.)
                            conn.send_bytes(pack(actions=clipped,normalized=normalized[0],clip_count=np.asarray(np.count_nonzero(clipped!=decoded))))
                            latencies.append(time.monotonic()-tick);queries+=1
                            if queries==1:np.savez_compressed(a.output/'first_live_query.npz',**data,normalized=normalized,decoded=decoded)
                            if queries%50==0:print('POLICY_QUERIES',queries,'mean_seconds',float(np.mean(latencies)),flush=True)
                    returncode=process.wait(timeout=120)
                    if returncode:raise RuntimeError(f'Simulator failed: {returncode}')
                finally:
                    if process.poll() is None:
                        process.terminate()
                        try:process.wait(timeout=15)
                        except subprocess.TimeoutExpired:process.kill();process.wait()
    report.update(queries=queries,mean_inference_seconds=float(np.mean(latencies)),status='complete')
    (a.output/'inference_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print('POLICY_EVALUATION_COMPLETE',flush=True)

if __name__=='__main__':main()
