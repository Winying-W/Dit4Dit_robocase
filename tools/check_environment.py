"""Import required training interfaces without loading weights or initializing CUDA."""
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments/robocasa_official_recipe/source'))
for module in ['torch', 'torchvision', 'diffusers', 'transformers', 'decord', 'av', 'pytorch3d.transforms',
               'DiT4DiT.dataloader.robocasa365_datasets', 'scripts.robocasa365.train_distributed']:
    importlib.import_module(module)
from diffusers import Cosmos2_5_PredictBasePipeline
from transformers import Qwen3VLForConditionalGeneration
print(json.dumps({name:importlib.metadata.version(name) for name in
                  ['torch','torchvision','diffusers','transformers','pytorch3d','eva-decord','pydantic']},indent=2))
print('Training imports passed. Model loading/GPU memory/NCCL still require runtime validation.')
