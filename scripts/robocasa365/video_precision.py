"""Keep precision identical across frozen/joint controls and checkpoint inference."""

def prepare_partial_video_fp32(model):
    # Training weights retain small updates in FP32. Both experimental arms and
    # their evaluation use this same precision; model forwards retain autocast.
    for index in (16, 17):
        model.backbone_interface.extractor.transformer.transformer_blocks[index].float()


def apply_saved_video_precision(model, payload):
    precision = payload.get('training_schedule', {}).get('selected_video_parameter_precision')
    if precision is None:
        return 'base_dtype'
    if precision != 'float32':
        raise ValueError(f'Unsupported saved video precision: {precision}')
    prepare_partial_video_fp32(model)
    return precision
