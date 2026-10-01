"""Versioned, opt-in scene construction; never edits the installed simulator.

stable_counter_v1 replaces unordered identity-hash deduplication of counter
geoms with first-occurrence order. Physics, actions and success are unchanged.
"""
import hashlib
import inspect
from copy import deepcopy
import os
from pathlib import Path
import textwrap


OFFICIAL = 'official_fresh_v1'
STABLE = 'stable_counter_v1'
PROTOCOLS = (OFFICIAL, STABLE)
COUNTER_SHA256 = '77f992d01aa1ae5f21ed7f170d0aab9c303c32148f4b427651c04912be8ce68a'
STABLE_FUNCTION_SHA256 = '44bd88ac94891a4df72203dfaffa47bc934256162f05b3c3ac2c978f6f70fe80'
ORIGINAL = 'valid_geoms = list(set(valid_geoms))'
REPLACEMENT = 'valid_geoms = list(dict.fromkeys(valid_geoms))'
_installed = None


def protocol_spec(name=OFFICIAL):
    if name not in PROTOCOLS:
        raise ValueError(f'Unknown scene protocol: {name}')
    recipe = None if name == OFFICIAL else dict(
        module='robocasa.models.fixtures.counter', method='Counter.get_reset_regions',
        source_sha256=COUNTER_SHA256, original=ORIGINAL, replacement=REPLACEMENT,
        effective_function_sha256=STABLE_FUNCTION_SHA256)
    return dict(name=name, counter_region_patch=recipe, python_hash_seed='0' if name == STABLE else None)


def simulator_environment(name, parent=None):
    """Hash seeding takes effect at interpreter startup, never via late mutation."""
    specification = protocol_spec(name)
    environment = dict(os.environ if parent is None else parent)
    if specification['python_hash_seed'] is not None:
        environment['PYTHONHASHSEED'] = specification['python_hash_seed']
    return environment


def _counter_module():
    import robocasa.models.fixtures.counter as module
    return module


def apply_scene_protocol(name=OFFICIAL):
    """Apply once per simulator process; repeated calls verify identity again."""
    global _installed
    specification = protocol_spec(name)
    if name == STABLE and os.environ.get('PYTHONHASHSEED') != '0':
        raise RuntimeError('stable_counter_v1 requires a fresh Python process started with PYTHONHASHSEED=0')
    module = _counter_module()
    path = Path(module.__file__).resolve()
    source_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    current = module.Counter.get_reset_regions
    if _installed is not None:
        applied_module, applied_function, receipt = _installed
        if module is not applied_module or current is not applied_function:
            raise RuntimeError('Scene protocol function changed after initialization')
        if receipt['protocol'] != specification or receipt['source_sha256'] != source_sha:
            raise RuntimeError('Cannot change scene protocol or counter source inside a simulator process')
        return deepcopy(receipt)
    source_file = inspect.getsourcefile(current)
    if source_file is None or Path(source_file).resolve() != path:
        raise RuntimeError('Counter reset function was already replaced before protocol initialization')
    source = textwrap.dedent(inspect.getsource(current))
    if name == STABLE:
        if source_sha != COUNTER_SHA256 or source.count(ORIGINAL) != 1:
            raise RuntimeError('Counter source differs from the reviewed stable_counter_v1 revision')
        source = source.replace(ORIGINAL, REPLACEMENT)
        if hashlib.sha256(source.encode()).hexdigest() != STABLE_FUNCTION_SHA256:
            raise RuntimeError('Counter effective function differs from the tested revision')
        namespace = {}
        exec(compile(source, '<stable_counter_v1>', 'exec'), current.__globals__, namespace)
        current = namespace['get_reset_regions']
        module.Counter.get_reset_regions = current
    receipt = dict(protocol=specification, source_path=str(path), source_sha256=source_sha,
                   python_hash_seed=os.environ.get('PYTHONHASHSEED'),
                   effective_function_sha256=hashlib.sha256(source.encode()).hexdigest(),
                   installed_file_unchanged=hashlib.sha256(path.read_bytes()).hexdigest() == source_sha)
    if not receipt['installed_file_unchanged']:
        raise RuntimeError('Installed counter source changed during protocol initialization')
    _installed = (module, current, receipt)
    return deepcopy(receipt)


def validate_protocol_records(identity, verification, evaluation):
    """Legacy records lacking this field mean the original behavior.

    Stable records require the simulator's receipt as well as the CLI label.
    """
    specs = [row.get('scene_protocol', protocol_spec()) for row in (identity, verification, evaluation)]
    for specification in specs:
        if not isinstance(specification, dict) or specification != protocol_spec(specification.get('name')):
            raise ValueError('Unknown or altered scene protocol recipe')
    if not specs[0] == specs[1] == specs[2]:
        raise ValueError('Scene protocol differs between benchmark, policy and simulator')
    runtime = evaluation.get('scene_protocol_runtime')
    if runtime is None:
        if specs[0]['name'] != OFFICIAL:
            raise ValueError('Stable scene protocol lacks simulator application receipt')
        return specs[0]
    if not isinstance(runtime, dict):
        raise ValueError('Invalid simulator scene protocol receipt')
    if runtime.get('protocol') != specs[0] or runtime.get('installed_file_unchanged') is not True:
        raise ValueError('Invalid simulator scene protocol receipt')
    if specs[0]['name'] == STABLE:
        recipe = specs[0]['counter_region_patch']
        if (runtime.get('source_sha256') != recipe['source_sha256'] or
                runtime.get('effective_function_sha256') != recipe['effective_function_sha256'] or
                runtime.get('python_hash_seed') != '0'):
            raise ValueError('Simulator counter function differs from the declared scene protocol')
    return specs[0]
