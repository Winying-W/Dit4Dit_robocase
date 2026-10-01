"""Verify the historical partial-video recipe on a real CPU model and demo.

This uses released GR1 initialization, not the missing original step2000. It
checks trainable modules and loss routing without taking an optimizer step.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from omegaconf import OmegaConf

from DiT4DiT.dataloader.gr00t_lerobot.schema import DatasetStatisticalValues
from DiT4DiT.dataloader.robocasa365_datasets import Robocasa365DatasetAdapter
from DiT4DiT.model.framework.base_framework import baseframework
from scripts.robocasa365.train_single_gpu import configure


def gradient_report(parameters):
    missing = [name for name, parameter in parameters.items() if parameter.grad is None]
    nonfinite = [name for name, parameter in parameters.items()
                 if parameter.grad is not None and not torch.isfinite(parameter.grad).all()]
    norm = math.sqrt(sum(float(parameter.grad.double().square().sum())
                         for parameter in parameters.values() if parameter.grad is not None))
    return dict(tensors=len(parameters), parameters=sum(p.numel() for p in parameters.values()),
                missing=missing, nonfinite=nonfinite, gradient_norm=norm)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True,
                        help='Verified original16-demo split/normalization/data_config directory')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if torch.cuda.is_available():
        raise RuntimeError('Use CUDA_VISIBLE_DEVICES empty for this CPU-only verification')
    if args.output.exists():
        raise FileExistsError('Use a new output directory to preserve previous evidence')
    args.output.mkdir(parents=True)
    torch.set_num_threads(8)
    torch.set_num_interop_threads(2)
    torch.manual_seed(123)
    started = time.monotonic()

    split = json.loads((args.prepared / 'split.json').read_text())
    config = OmegaConf.load(args.prepared / 'data_config.yaml')
    statistics_path = args.prepared / 'normalization.json'
    statistics_hash = hashlib.sha256(statistics_path.read_bytes()).hexdigest()
    assert statistics_hash == '77d32db8bb1b30e24ec7680585858118e4bda30a6f3102eb2392ef7d1011408a'
    statistics = json.loads(statistics_path.read_text())
    dataset_path = Path(config.dataset_path)
    dataset = Robocasa365DatasetAdapter(dataset_path, config)
    modality = json.loads((dataset_path / 'meta/modality.json').read_text())
    for group, column in [('action', 'action'), ('state', 'observation.state')]:
        for name, description in modality[group].items():
            values = {key: value[description['start']:description['end']]
                      for key, value in statistics[column].items()}
            getattr(dataset.dataset.metadata.statistics, group)[name] = (
                DatasetStatisticalValues.model_validate(values))
    dataset.dataset.transforms.set_metadata(dataset.dataset.metadata)
    index = next(index for index, (episode, step) in enumerate(dataset.dataset.all_steps)
                 if int(episode) == 5 and int(step) == 100)
    assert 5 in split['train_episodes']
    example = dataset[index]
    assert len(example['image']) == 9
    print('CPU_PARTIAL_VIDEO_BATCH_READY', index, flush=True)

    model = baseframework.from_pretrained(str(args.checkpoint))
    model.config.datasets.vla_data = config
    model.config.trainer.repeated_diffusion_steps = 1
    trainable = configure(model, 'partial_joint')
    assert all(parameter.device.type == 'cpu' for parameter in model.parameters())
    action = {name: parameter for name, parameter in trainable.items()
              if name.startswith('action_model.')}
    video = {name: parameter for name, parameter in trainable.items()
             if name.startswith('backbone_interface.extractor.transformer.transformer_blocks.')}
    assert len(action) == 247 and set(action) | set(video) == set(trainable)
    blocks = {name: parameter for name, parameter in video.items()
              if any(name.startswith(f'backbone_interface.extractor.transformer.transformer_blocks.{i}.')
                     for i in (16, 17))}
    assert video and set(blocks) == set(video)
    assert not any(parameter.requires_grad for parameter in model.backbone_interface.extractor.text_encoder.parameters())
    assert not any(parameter.requires_grad for parameter in model.backbone_interface.extractor.vae.parameters())
    print('CPU_PARTIAL_VIDEO_FORWARD_START', len(action), len(video), flush=True)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        losses = model(examples=[example])
    values = {name: float(loss.detach()) for name, loss in losses.items()}
    assert all(math.isfinite(value) for value in values.values())
    assert losses['action_loss'].requires_grad and losses['future_video_loss'].requires_grad

    # Separate backwards prove the released implementation's stop-gradient route.
    losses['action_loss'].backward()
    action_gradients = gradient_report(action)
    assert not action_gradients['missing'] and not action_gradients['nonfinite']
    assert action_gradients['gradient_norm'] > 0
    assert all(parameter.grad is None for parameter in video.values())
    print('CPU_ACTION_LOSS_ROUTE_PASSED', flush=True)

    model.zero_grad(set_to_none=True)
    losses['future_video_loss'].backward()
    video_gradients = gradient_report(video)
    assert not video_gradients['missing'] and not video_gradients['nonfinite']
    assert video_gradients['gradient_norm'] > 0
    assert all(parameter.grad is None for parameter in action.values())
    per_block = {}
    for block in (16, 17):
        prefix = f'backbone_interface.extractor.transformer.transformer_blocks.{block}.'
        per_block[str(block)] = gradient_report({name: parameter for name, parameter in video.items()
                                                 if name.startswith(prefix)})
        assert per_block[str(block)]['gradient_norm'] > 0
    assert all(parameter.grad is None for parameter in model.parameters() if not parameter.requires_grad)
    sources = ['scripts/robocasa365/verify_partial_video_cpu.py',
               'scripts/robocasa365/train_single_gpu.py',
               'DiT4DiT/model/framework/DiT4DiT.py',
               'DiT4DiT/model/modules/vlm/Cosmos25.py']
    report = dict(status='complete', verified_utc=datetime.now(timezone.utc).isoformat(),
                  device='cpu', initialization='released_GR1_not_original_trained_step2000',
                  checkpoint=str(args.checkpoint.resolve()),
                  dataset=str(dataset_path), episode=5, timestep=100,
                  video_frames=len(example['image']), normalization_sha256=statistics_hash,
                  strict_released_checkpoint_load=True, losses=values,
                  action_loss_gradients=action_gradients, video_loss_gradients=video_gradients,
                  video_gradient_by_block=per_block,
                  action_loss_updates_video=False, video_loss_updates_action=False,
                  frozen_text_and_vae_have_no_gradients=True, optimizer_updates=0,
                  elapsed_seconds=time.monotonic() - started,
                  source_sha256={name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in sources},
                  scope='Real CPU forward and separate backwards with a real training demo; '
                        'verifies only the planned partial-video recipe and gradient routes. '
                        'Does not recover original trained weights, execute the historical A/B, '
                        'certify GPU memory, or measure policy success.')
    (args.output / 'acceptance.json').write_text(json.dumps(report, indent=2) + '\n')
    print('CPU_PARTIAL_VIDEO_PASSED', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
