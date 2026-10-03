"""Temporarily enter inference-safe validation without changing training precision."""
from contextlib import contextmanager


@contextmanager
def joint_evaluation_mode(model):
    video=model.backbone_interface.extractor.transformer
    checkpointing=video.gradient_checkpointing
    modules=[(module,module.training) for module in model.modules()]
    parameters=[(parameter,parameter.requires_grad) for parameter in model.parameters()]
    model.eval();model.requires_grad_(False);video.disable_gradient_checkpointing()
    try:
        yield
    finally:
        for parameter,requires_grad in parameters:parameter.requires_grad_(requires_grad)
        for module,training in modules:module.training=training
        if checkpointing:video.enable_gradient_checkpointing()
