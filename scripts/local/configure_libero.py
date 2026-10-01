"""Create a project-scoped LIBERO configuration without interactive prompts."""
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[2]
base = ROOT / 'third_party/LIBERO/libero/libero'
config_dir = ROOT / '.cache/libero'
config_dir.mkdir(parents=True, exist_ok=True)
data = ROOT / '.cache/libero/datasets'
data.mkdir(exist_ok=True)
paths = {'benchmark_root': base, 'bddl_files': base/'bddl_files', 'init_states': base/'init_files', 'assets': base/'assets', 'datasets': data}
for key, path in paths.items():
    if not path.is_dir():
        raise FileNotFoundError(f'{key}: {path}')
(config_dir/'config.yaml').write_text(yaml.safe_dump({k:str(v) for k,v in paths.items()}))
print('LIBERO configuration ready:', config_dir/'config.yaml')

# Avoid a shared /tmp/robosuite.log owned by another server user.
from importlib.util import find_spec
robosuite_root=Path(next(iter(find_spec('robosuite').submodule_search_locations)))
(robosuite_root/'macros_private.py').write_text(
    'from robosuite.macros import *\n'
    'import robosuite.macros as _macros\n'
    '_macros.FILE_LOGGING_LEVEL = None\n'
    'FILE_LOGGING_LEVEL = None\n'
)
# The official bundled initial states contain NumPy objects. PyTorch 2.6+
# changed the default to weights_only=True; these are trusted repository files.
benchmark_file=base/'benchmark/__init__.py'
text=benchmark_file.read_text()
text=text.replace('torch.load(init_states_path)', 'torch.load(init_states_path, weights_only=False)')
benchmark_file.write_text(text)
