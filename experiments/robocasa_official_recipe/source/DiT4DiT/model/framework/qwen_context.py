"""Optional frozen Qwen tokens; the existing Cosmos context is unchanged."""
import numpy as np
import torch
from torch import nn


class QwenProjection(nn.Module):
    def __init__(self, input_dim=2560, output_dim=2048):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.linear = nn.Linear(input_dim, output_dim)  # Standard PyTorch random initialization.

    def forward(self, cosmos, qwen, valid):
        q = self.linear(self.norm(qwen.detach().float())).to(cosmos.dtype)
        context = torch.cat((cosmos, q), dim=1)
        keep = torch.cat((torch.ones(cosmos.shape[:2], dtype=torch.bool, device=cosmos.device), valid.bool()), dim=1)
        mask = torch.zeros(keep.shape, dtype=context.dtype, device=context.device).masked_fill(~keep, float('-inf'))
        return context, mask[:, None, :]


def current_views(example):
    value = example['image'][0]
    pixels = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    if pixels.ndim != 3:
        raise ValueError(f'Expected current RGB mosaic, got {pixels.shape}')
    if pixels.shape[0] == 3 and pixels.shape[-1] != 3:
        pixels = pixels.transpose(1, 2, 0)
    if pixels.shape[-1] != 3 or pixels.shape[1] != 3*pixels.shape[0]:
        raise ValueError(f'Expected three horizontal square views, got {pixels.shape}')
    if pixels.dtype != np.uint8:
        pixels = np.rint(np.clip(pixels, 0, 1)*255).astype(np.uint8) if pixels.max() <= 1 else np.rint(np.clip(pixels, 0, 255)).astype(np.uint8)
    return list(np.split(pixels, 3, axis=1))


class QwenContext(nn.Module):
    def __init__(self, path, seed=42):
        super().__init__()
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        # Adding this branch must not advance the baseline's action/video RNG.
        devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            full, info = Qwen3VLForConditionalGeneration.from_pretrained(
                path, dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True, output_loading_info=True)
            if info['missing_keys'] or info['unexpected_keys'] or info.get('mismatched_keys'):
                raise ValueError(f'Qwen checkpoint loading mismatch: {info}')
            self.encoder = full.model.requires_grad_(False).eval()
            self.projection = QwenProjection()
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        self.processor.image_processor.size = dict(shortest_edge=224**2, longest_edge=224**2)
        self.processor.tokenizer.padding_side = 'right'
        self.enabled = True
        self.settings = dict(path=str(path), seed=seed, image_pixels=224**2, initialization='torch_default_random', gain=None)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, cosmos, examples):
        if not self.enabled:
            return cosmos, None
        messages = [[dict(role='user', content=[*[dict(type='image', image=im) for im in current_views(ex)], dict(type='text', text=ex['lang'])])] for ex in examples]
        inputs = self.processor.apply_chat_template(messages, tokenize=True, padding=True, add_generation_prompt=True,
            return_dict=True, return_tensors='pt').to(cosmos.device)
        # Isolate any encoder-side randomness from the baseline action/video streams.
        devices = [cosmos.device.index] if cosmos.is_cuda else []
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            encoded = self.encoder(**inputs, use_cache=False, output_hidden_states=False, return_dict=True).last_hidden_state
        return self.projection(cosmos, encoded, inputs['attention_mask'])


def attach_qwen(model, path, seed=42):
    if path:
        model.qwen_context = QwenContext(str(path), seed)
    return model
