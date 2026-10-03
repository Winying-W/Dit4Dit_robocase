"""Convert public NVIDIA-format tensors using a pinned upstream Diffusers mapping."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import torch
from safetensors.torch import load_file
from accelerate import init_empty_weights
from diffusers import CosmosTransformer3DModel, AutoencoderKLWan, UniPCMultistepScheduler
from diffusers.pipelines.cosmos.pipeline_cosmos2_5_predict import Cosmos2_5_PredictBasePipeline
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--vae',type=Path,required=True);p.add_argument('--tokenizer',type=Path,required=True);p.add_argument('--converter',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--official-manifest',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Use a fresh output to keep conversion atomic')
    upstream=json.loads((a.source/'upstream_manifest.json').read_text())
    f=a.source/'transformer/cosmos2_5_post_trained.safetensors'
    expected=next(x for x in upstream['siblings'] if x['rfilename']==str(f.relative_to(a.source)))['lfs']['sha256']
    assert sha(f)==expected
    official={r['rfilename']:r for r in json.loads(a.official_manifest.read_text())['siblings']}
    for name,record in official.items():
        if name.startswith('text_encoder/') and name.endswith('.safetensors'):
            assert sha(a.source/name)==record['lfs']['sha256'],name
    spec=importlib.util.spec_from_file_location('upstream_cosmos_conversion',a.converter);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    state=load_file(f)
    mapped={}
    for name,tensor in state.items():
        name=name.removeprefix('net.')
        for old,new in module.TRANSFORMER_KEYS_RENAME_DICT_COSMOS_2_0.items():name=name.replace(old,new)
        if any(k in name for k in module.TRANSFORMER_SPECIAL_KEYS_REMAP_COSMOS_2_0):continue
        assert name not in mapped,name;mapped[name]=tensor
    with init_empty_weights():model=CosmosTransformer3DModel(**module.TRANSFORMER_CONFIGS['Cosmos-2.5-Predict-Base-2B'])
    expected_state=model.state_dict()
    assert set(expected_state)==set(mapped),dict(missing=sorted(set(expected_state)-set(mapped)),extra=sorted(set(mapped)-set(expected_state)))
    assert all(expected_state[k].shape==mapped[k].shape for k in mapped)
    model.load_state_dict(mapped,strict=True,assign=True);del mapped,state,expected_state
    vae=AutoencoderKLWan.from_pretrained(str(a.vae),torch_dtype=torch.float32,local_files_only=True)
    text=Qwen2_5_VLForConditionalGeneration.from_pretrained(str(a.source/'text_encoder'),torch_dtype='auto',local_files_only=True)
    tokenizer=AutoTokenizer.from_pretrained(str(a.tokenizer),local_files_only=True)
    scheduler=UniPCMultistepScheduler(use_karras_sigmas=True,use_flow_sigmas=True,prediction_type='flow_prediction',sigma_max=200.,sigma_min=.01)
    pipe=Cosmos2_5_PredictBasePipeline(text_encoder=text,tokenizer=tokenizer,transformer=model,vae=vae,scheduler=scheduler,safety_checker=lambda *args,**kwargs:None)
    temp=a.output.with_name(a.output.name+'.converting');temp.mkdir(parents=True,exist_ok=True)
    pipe.save_pretrained(temp,safe_serialization=True,max_shard_size='5GB')
    report=dict(source_repo=upstream['id'],source_revision=upstream['sha'],source_transformer_sha256=expected,converter_sha256=sha(a.converter),
        converter_revision='3996788b602eaae4da41a1d45726b62e662b73cf',transformer_mapping='Official Cosmos 2.5 rename/remove mapping; exact key/shape and strict load verified',
        transformer_weights='NVIDIA Cosmos 2.5 post-trained 2B public-source tensors; not GR1 policy weights',
        text_encoder='Public source shards SHA256 matched official Cosmos Diffusers metadata in source audit',
        vae_source=str(a.vae.resolve()),tokenizer_source=str(a.tokenizer.resolve()),files={})
    for file in sorted(temp.rglob('*')):
        if file.is_file():report['files'][str(file.relative_to(temp))]=dict(bytes=file.stat().st_size,sha256=sha(file))
    official={f['rfilename']:f for f in json.loads(a.official_manifest.read_text())['siblings']}
    report['official_weight_hash_matches']={}
    for name in ['transformer/diffusion_pytorch_model.safetensors','vae/diffusion_pytorch_model.safetensors']:
        match=report['files'][name]['sha256']==official[name]['lfs']['sha256']
        report['official_weight_hash_matches'][name]=match
    (temp/'CONVERSION_VERIFIED.json').write_text(json.dumps(report,indent=2))
    if not all(report['official_weight_hash_matches'].values()):
        raise RuntimeError('Converted weight files differ from the official manifest; investigate serialization/tensor identity before using them')
    temp.rename(a.output)
    print('CONVERSION_COMPLETE',a.output,flush=True)


if __name__=='__main__':main()
