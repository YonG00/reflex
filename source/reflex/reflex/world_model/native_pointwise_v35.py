"""Opaque native FP32 pointwise ops for opt-in compilation diagnostics.

Forward and backward both call the original ATen math. This retains an external
kernel boundary inside a full graph; no detach and no surrogate derivative.
Normalization statistics are fixed buffers, exactly as Standardizer specifies.
"""

import torch


@torch.library.custom_op("reflex_v35::silu_backward_native", mutates_args=())
def silu_backward_native(gradient: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    return torch.ops.aten.silu_backward.default(gradient, value)


@silu_backward_native.register_fake
def _(gradient, value):
    return torch.empty_like(value)


@torch.library.custom_op("reflex_v35::silu_native", mutates_args=())
def silu_native(value: torch.Tensor) -> torch.Tensor:
    return torch.ops.aten.silu.default(value)


@silu_native.register_fake
def _(value):
    return torch.empty_like(value)


def silu_setup(ctx, inputs, output):
    ctx.save_for_backward(inputs[0])


def silu_backward(ctx, gradient):
    return silu_backward_native(gradient, ctx.saved_tensors[0])


silu_native.register_autograd(silu_backward, setup_context=silu_setup)


@torch.library.custom_op("reflex_v35::division_native", mutates_args=())
def division_native(value: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    if std.requires_grad:
        raise ValueError("V35 native division supports fixed statistics only")
    return torch.ops.aten.div.Tensor(value, std)


@division_native.register_fake
def _(value, std):
    return torch.empty_like(value)


def division_setup(ctx, inputs, output):
    if inputs[1].requires_grad:
        raise ValueError("Cannot discard normalization statistic gradients")
    ctx.save_for_backward(inputs[1])


def division_backward(ctx, gradient):
    return division_native(gradient, ctx.saved_tensors[0]), None


division_native.register_autograd(division_backward, setup_context=division_setup)


def norm_native(value, mean, std):
    if mean.requires_grad or std.requires_grad:
        raise ValueError("V35 native normalization supports fixed statistics only")
    # Keep the framework's own clamp forward/backward, including its exact
    # version-specific convention at +/-20. Fence only division rounding.
    return division_native(value - mean, std).clamp(-20, 20)
