"""RoboCasa365 LeRobot adapter; camera axis is explicitly separate from time.

State/actions retain the dataset modality order (including base and control mode).
The stock DiT4DiT model consumes a list of temporally ordered RGB mosaics.
"""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from DiT4DiT.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset, ModalityConfig
from DiT4DiT.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from DiT4DiT.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from DiT4DiT.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform


class Robocasa365RobotConfig:
    camera_names = ('robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand')
    video_keys = tuple('video.' + k for k in camera_names)
    state_keys = tuple('state.' + k for k in ('base_position', 'base_rotation',
        'end_effector_position_relative', 'end_effector_rotation_relative', 'gripper_qpos'))
    action_keys = tuple('action.' + k for k in ('base_motion', 'control_mode',
        'end_effector_position', 'end_effector_rotation', 'gripper_close'))
    language_keys = ('annotation.human.task_description',)
    state_dim, action_dim = 16, 12
    dataset_to_env_action = (5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4)

    def __init__(self, horizon=16):
        self.horizon = horizon

    def modality_config(self):
        return {name: ModalityConfig(delta_indices=list(range(self.horizon)) if name == 'action' else [0],
                modality_keys=list(keys)) for name, keys in (
                    ('video', self.video_keys), ('state', self.state_keys),
                    ('action', self.action_keys), ('language', self.language_keys))}

    def transform(self):
        # Quaternions are already bounded; do not apply GR1 joint-angle sin/cos.
        # Binary commands retain their official values. Continuous actions use dataset min/max.
        continuous = [k for k in self.action_keys if k not in ('action.control_mode', 'action.gripper_close')]
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=list(self.state_keys + self.action_keys)),
            StateActionTransform(apply_to=continuous, normalization_modes={k:'min_max' for k in continuous}),
        ])


def camera_mosaics(frames, camera_keys, image_size=(224,224)):
    """Return [T] CHW float RGB images, each containing all cameras at the same t."""
    tensors = []
    for key in camera_keys:
        rgb = np.asarray(frames[key])
        if rgb.ndim != 4 or rgb.shape[-1] != 3:
            raise ValueError(f'{key}: expected T,H,W,3 RGB, got {rgb.shape}')
        x = torch.as_tensor(rgb.copy()).permute(0,3,1,2).float() / 255.0
        tensors.append(F.interpolate(x, size=image_size, mode='bilinear', align_corners=False))
    if len({x.shape[0] for x in tensors}) != 1:
        raise ValueError('Camera timestamps must have matching lengths')
    return list(torch.cat(tensors, dim=-1).unbind(0))


class Robocasa365DatasetAdapter(Dataset):
    def __init__(self, dataset_path, data_cfg):
        self.cfg = data_cfg
        self.robot = Robocasa365RobotConfig(int(data_cfg.get('action_horizon', 16)))
        self.dataset = LeRobotSingleDataset(dataset_path=Path(dataset_path),
            modality_configs=self.robot.modality_config(), transforms=self.robot.transform(),
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT, data_cfg=data_cfg,
            video_backend=data_cfg.get('video_backend', 'decord'),
            video_backend_kwargs={'num_threads':1} if data_cfg.get('video_backend','decord') == 'decord' else {})
        self.state_dim = int(data_cfg.get('max_state_dim',64))
        self.action_dim = int(data_cfg.get('max_action_dim',32))
        if self.state_dim < 16 or self.action_dim < 12:
            raise ValueError('Full RoboCasa365 schema requires state >=16 and action >=12')

    def decode_actions(self, normalized, env_order=True):
        """Undo target-dataset normalization and optionally produce robosuite order.

        Keep binary mode/gripper continuous here; thresholding belongs to a future
        policy execution protocol, not this interface acceptance check.
        """
        action = np.asarray(normalized, dtype=np.float32)[..., :12].copy()
        if action.shape[-1] != 12:
            raise ValueError("Expected at least 12 action dimensions")
        offset = 0
        for key, width in zip(self.robot.action_keys, (4, 1, 3, 3, 1)):
            if key not in ("action.control_mode", "action.gripper_close"):
                stats = self.dataset.metadata.statistics.action[key.removeprefix("action.")]
                low, high = np.asarray(stats.min), np.asarray(stats.max)
                action[..., offset:offset+width] = (action[..., offset:offset+width]+1)/2*(high-low)+low
            offset += width
        if env_order:
            action = action[..., list(self.robot.dataset_to_env_action)]
        return action

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        episode, step = self.dataset.all_steps[index]
        raw = self.dataset.get_step_data(episode, step)
        images = camera_mosaics(raw, self.robot.video_keys, tuple(self.cfg.get('image_size',[224,224])))
        ratio = int(self.cfg.get('action_video_freq_ratio',2))
        images = images[::ratio]
        data = self.dataset.transforms(raw)
        state = np.concatenate([np.asarray(data[k]) for k in self.robot.state_keys], axis=-1).astype(np.float32)
        action = np.concatenate([np.asarray(data[k]) for k in self.robot.action_keys], axis=-1).astype(np.float32)
        if state.shape != (1,16) or action.shape != (self.robot.horizon,12):
            raise ValueError(f'Unexpected state/action shapes: {state.shape}/{action.shape}')
        length = self.dataset.trajectory_lengths[self.dataset.get_trajectory_index(episode)]
        valid_time = np.arange(self.robot.horizon) + step < length
        action_mask = np.zeros((self.robot.horizon,self.action_dim),dtype=bool)
        action_mask[:,:12] = valid_time[:,None]
        state_mask = np.zeros((1,self.state_dim),dtype=bool)
        state_mask[:,:16] = True
        return dict(image=images, lang=data[self.robot.language_keys[0]][0],
            state=np.pad(state,((0,0),(0,self.state_dim-16))),
            action=np.pad(action,((0,0),(0,self.action_dim-12))),
            state_mask=state_mask, action_mask=action_mask)


def collate_fn(batch):
    # Stock DiT4DiT forward takes List[dict], not a pre-stacked dictionary.
    return batch


def get_vla_dataset(data_cfg, **kwargs):
    return Robocasa365DatasetAdapter(data_cfg.dataset_path, data_cfg)
