"""Run every declared task through the real simulator before GPU submission."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from scripts.robocasa365.scene_protocol import OFFICIAL, PROTOCOLS, protocol_spec, simulator_environment, validate_protocol_records


def write(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def validate_environment_report(report, manifest, scene_protocol, rendering_enabled):
    """Refuse mixed protocols, missing tasks and CPU evidence passed as rendering."""
    declaration=dict(scene_protocol=protocol_spec(scene_protocol))
    assert report['status']=='complete' and report['task_set']==manifest['task_set']
    assert report['scene_protocol']==declaration['scene_protocol']
    assert report['rendering_enabled'] is rendering_enabled
    assert [row['task'] for row in report['tasks']]==[row['task'] for row in manifest['tasks']]
    for row,task in zip(report['tasks'],manifest['tasks']):
        validate_protocol_records(declaration,report,row)
        assert row['passed'] and row['returncode']==0 and row['action_chain_runtime_passed']
        assert row['steps']>=8 and row['horizon']==task['horizon']
        assert row['controller']['action_dim']==12 and row['controller']['input_type']=='delta'
        assert row['rendering_enabled'] is rendering_enabled
        deviations=row['image_std_by_camera']
        assert len(deviations)==3
        assert all(value>0 for value in deviations) if rendering_enabled else all(value==0 for value in deviations)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--sim-python',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=8)
    parser.add_argument('--scene-protocol',choices=PROTOCOLS,default=OFFICIAL)
    parser.add_argument('--no-render',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.steps<8:raise ValueError('Use at least eight full-schema random steps per task')
    manifest=json.loads(args.manifest.read_text());rows=[]
    report=dict(status='running',manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        task_set=manifest['task_set'],tasks=rows,scene_protocol=protocol_spec(args.scene_protocol),
        rendering_enabled=not args.no_render,
        scope=('All declared task environments: CPU reset, full12D step and controller with black image placeholders; '
               'no rendering or policy success trials' if args.no_render else
               'All declared task environments: reset, full12D step, controller and three-camera video; no policy success trials'))
    for name in ['acceptance.json','progress.json']:
        if (args.output/name).exists():
            raise FileExistsError(f'Use a new output directory; prior evidence exists at {args.output/name}')
    environment=simulator_environment(args.scene_protocol)
    if args.no_render:environment['CUDA_VISIBLE_DEVICES']=''
    for task in manifest['tasks']:
        folder=args.output/task['task'];folder.mkdir(exist_ok=True)
        command=[args.sim_python,'scripts/robocasa365/env_smoke.py','--task',task['task'],
            '--output',str(folder),'--steps',str(args.steps),'--allow-mobile','--scene-protocol',args.scene_protocol]
        if args.no_render:command.append('--no-render')
        with (folder/'environment.log').open('w') as stream:
            completed=subprocess.run(command,env=environment,stdout=stream,stderr=subprocess.STDOUT,timeout=300)
        if completed.returncode:
            report.update(status='failed',failed_task=task['task'],returncode=completed.returncode)
            write(args.output/'acceptance.json',report)
            raise RuntimeError(f'Environment failed for {task["task"]}; inspect {folder}')
        row=json.loads((folder/'env.json').read_text())
        assert row['task']==task['task'] and row['horizon']==task['horizon']
        assert row['passed'] and row['action_chain_runtime_passed'] and row['steps']>=args.steps
        validate_protocol_records(report,report,row)
        assert row['rendering_enabled'] is (not args.no_render)
        if not args.no_render:assert (folder/'random_rollout.mp4').stat().st_size>0
        row['returncode']=completed.returncode;rows.append(row)
        write(args.output/'progress.json',report)
        print('ENVIRONMENT_VERIFIED',task['task'],row['steps'],flush=True)
    report['status']='complete'
    validate_environment_report(report,manifest,args.scene_protocol,not args.no_render)
    write(args.output/'acceptance.json',report)


if __name__=='__main__':main()
