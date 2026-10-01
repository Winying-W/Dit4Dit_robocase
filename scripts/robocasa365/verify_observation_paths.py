"""Compare real dataset observations through training and live-policy input paths.

This is a CPU input-interface audit, not a model forward or simulator rollout.
It uses recorded RGB for both paths and cannot prove fresh-scene alignment.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import torch

from DiT4DiT.dataloader.robocasa365_datasets import Robocasa365DatasetAdapter, camera_mosaics
from DiT4DiT.model.modules.vlm.Cosmos25 import _Cosmos25_Interface
from scripts.robocasa365.eval_protocol import CAMERAS, STATE_KEYS, observation_arrays, pack, unpack
from scripts.robocasa365.multitask_data import apply_statistics


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def live_builder(source):
    """Execute the production frames/ex assignments, rather than copy their logic."""
    tree=ast.parse(Path(source).read_text())
    selected=[]
    for name in ['frames','ex']:
        candidates=[node for node in ast.walk(tree) if isinstance(node,ast.Assign) and
                    len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id==name]
        if len(candidates)!=1:
            raise ValueError(f'Production assignment {name} is ambiguous; review the audit')
        selected.append(candidates[0])
    fragment=ast.fix_missing_locations(ast.Module(body=selected,type_ignores=[]))
    identity=hashlib.sha256(ast.dump(fragment,include_attributes=False).encode()).hexdigest()
    code=compile(fragment,str(source),'exec')
    def build(payload, adapter, config):
        namespace=dict(np=np,camera_mosaics=camera_mosaics,ds=adapter,cfg=config,
                       images=payload['images'],state=payload['state'],lang=str(payload['language']))
        exec(code,namespace)
        return namespace['ex']
    return build,identity


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifests',type=Path,nargs='+',required=True)
    parser.add_argument('--data-config',type=Path,required=True)
    parser.add_argument('--comparison-source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=9177)
    args=parser.parse_args()
    if torch.cuda.is_available() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise RuntimeError('This audit requires CUDA_VISIBLE_DEVICES empty')
    torch.set_num_threads(2)
    args.output.mkdir(parents=True,exist_ok=False)
    source=Path('scripts/robocasa365/evaluate_policy.py')
    build,identity=live_builder(source)
    _,comparison_identity=live_builder(args.comparison_source/source)
    assert identity==comparison_identity, 'Atomic and Composite live input expressions differ'
    dependencies=['DiT4DiT/dataloader/robocasa365_datasets.py','DiT4DiT/model/modules/vlm/Cosmos25.py',
                  'scripts/robocasa365/eval_protocol.py','scripts/robocasa365/multitask_data.py']
    for name in dependencies:
        assert digest(name)==digest(args.comparison_source/name),name
    report=dict(status='running',created_utc=datetime.now(timezone.utc).isoformat(),
                sample_seed=args.seed,selection='One seeded-random train and one validation episode per task; first, middle and last frame',
                production_input_ast_sha256=identity,comparison_input_ast_sha256=comparison_identity,
                source_sha256={name:digest(name) for name in [str(source),__file__,*dependencies]},
                manifests={str(p.resolve()):digest(p) for p in args.manifests},
                windows=[],tasks=[],learned_policy_trials=0,model_forward_executed=False,rendering_enabled=False,
                scope='Recorded RGB/state/instruction passed through actual dataset adapter, simulator observation serializer, '
                      'live evaluator input expressions and actual Cosmos input builder. Same recorded images on both paths; '
                      'not fresh-scene RGB alignment, neural prediction, training or policy success.')
    rng=np.random.default_rng(args.seed)
    for manifest_path in args.manifests:
        manifest=json.loads(manifest_path.read_text())
        statistics=json.loads((manifest_path.parent/'normalization.json').read_text())
        for task in manifest['tasks']:
            config=OmegaConf.load(args.data_config)
            config.statistics_path=str((manifest_path.parent/'normalization.json').resolve())
            dataset=Robocasa365DatasetAdapter(task['path'],config)
            apply_statistics(dataset,task['path'],statistics)
            assert tuple('video.'+name for name in CAMERAS)==dataset.robot.video_keys
            assert tuple('state.'+name for name in STATE_KEYS)==dataset.robot.state_keys
            modality=json.loads((Path(task['path'])/'meta/modality.json').read_text())
            selected={split:int(rng.choice(task[split+'_episodes'])) for split in ['train','validation']}
            indices={(int(ep),int(step)):i for i,(ep,step) in enumerate(dataset.dataset.all_steps) if int(ep) in selected.values()}
            files={int(p.stem.split('_')[-1]):p for p in (Path(task['path'])/'data').glob('*/episode_*.parquet')}
            for split,episode in selected.items():
                frame=pd.read_parquet(files[episode],columns=['observation.state'])
                positions=[0,len(frame)//2,len(frame)-1]
                assert len(set(positions))==3
                for step in positions:
                    training=dataset[indices[episode,step]]
                    training['image']=training['image'][:1]
                    raw=dataset.dataset.get_step_data(episode,step)
                    state=np.asarray(frame.iloc[step]['observation.state'],dtype=np.float32)
                    observation={'state.'+key:state[modality['state'][key]['start']:modality['state'][key]['end']].copy()
                                 for key in STATE_KEYS}
                    observation.update({'video.'+key:np.asarray(raw['video.'+key])[0].copy() for key in CAMERAS})
                    observation['annotation.human.task_description']=raw[dataset.robot.language_keys[0]][0]
                    payload=unpack(pack(**observation_arrays(observation)))
                    live=build(payload,dataset,config)
                    assert training['lang']==live['lang'] and isinstance(live['lang'],str)
                    assert training['state'].dtype==live['state'].dtype==np.float32
                    np.testing.assert_array_equal(training['state'],live['state'])
                    assert np.isfinite(live['state']).all() and not live['state'][:,16:].any()
                    assert len(training['image'])==len(live['image'])==1
                    torch.testing.assert_close(training['image'][0],live['image'][0],rtol=0,atol=0)
                    inputs=[]
                    for example in [training,live]:
                        inputs.append(_Cosmos25_Interface.build_cosmos_inputs(_Cosmos25_Interface,
                                      images=[example['image']],instructions=[example['lang']]))
                    torch.testing.assert_close(inputs[0]['videos'],inputs[1]['videos'],rtol=0,atol=0)
                    assert inputs[0]['prompts']==inputs[1]['prompts']
                    assert inputs[0]['future_videos'] is None and inputs[1]['future_videos'] is None
                    swapped=dict(payload,images=payload['images'][[1,0,2]])
                    swap_error=float((build(swapped,dataset,config)['image'][0]-training['image'][0]).abs().max())
                    assert swap_error>0, 'Camera-order negative control could not distinguish this observation'
                    reversed_state=build(dict(payload,state=payload['state'][::-1].copy()),dataset,config)['state']
                    assert not np.array_equal(reversed_state,training['state']), 'State-order negative control is degenerate'
                    report['windows'].append(dict(task=task['task'],task_set=manifest['task_set'],split=split,
                        episode=episode,step=step,frames=len(frame),state_max_error=0.,mosaic_max_error=0.,
                        cosmos_video_max_error=0.,cosmos_video_shape=list(inputs[0]['videos'].shape),
                        language_equal=True,raw_camera_dtype=str(payload['images'].dtype),
                        camera_swap_max_error=swap_error,state_order_negative_control_passed=True))
            report['tasks'].append(dict(task=task['task'],task_set=manifest['task_set'],selected_episodes=selected,windows=6))
            save(args.output/'progress.json',report)
            print('INPUT_PATH_VERIFIED',manifest['task_set'],task['task'],6,flush=True)
            del dataset
    report.update(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),
                  task_count=len(report['tasks']),episodes=2*len(report['tasks']),window_count=len(report['windows']))
    save(args.output/'report.json',report)
    print(json.dumps({key:report[key] for key in ['status','task_count','episodes','window_count','learned_policy_trials']}),flush=True)


if __name__=='__main__':main()
