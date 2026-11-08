"""Diagnostic FP32 compilation boundaries; no training/checkpoint changes.

All candidates use the existing features, Standardizer, MLP and integrate.
The proposed numerical fence keeps encoding/normalization/integration eager,
compiling only the MLP. Layerwise is a control for activation/fusion rounding.
"""

import torch
from torch import nn

from .physics import integrate

DRIVERS = (
    "wm_features",
    "wm_encoder",
    "wm_normalization",
    "wm_mlp",
    "wm_integrate",
    "wm_encoder_mlp",
    "wm_mlp_integrate",
    "wm_layerwise",
)


def decoder(model):
    def decode(state, delta, context):
        if model.com_indices is None:
            return integrate(state, delta, model.dt)
        root, box = model.com_indices
        return integrate(
            state,
            delta,
            model.dt,
            context[..., root : root + 3],
            context[..., box : box + 3],
        )

    return decode


def components(model):
    def encoder(state, target, context, event):
        return model.inputs(model.features(state, target, context, event))

    return {
        "features": model.features,
        "normalization": model.inputs,
        "mlp": model.net,
        "integrate": decoder(model),
        "encoder": encoder,
    }


def build(model, driver, backend="inductor"):
    """Return a Python callable, never register duplicate model submodules."""
    if driver not in DRIVERS:
        raise ValueError(f"Unknown Physical WM boundary: {driver}")
    c = components(model)
    compile_fn = lambda fn: torch.compile(
        fn, fullgraph=True, dynamic=False, backend=backend
    )
    selected = {
        "wm_features": "features",
        "wm_encoder": "encoder",
        "wm_normalization": "normalization",
        "wm_mlp": "mlp",
        "wm_integrate": "integrate",
    }
    if driver in selected:
        key = selected[driver]
        c[key] = compile_fn(c[key])
    if driver == "wm_layerwise":
        layers = [
            compile_fn(layer) if isinstance(layer, nn.Linear) else layer
            for layer in model.net
        ]

        def mlp(value):
            for layer in layers:
                value = layer(value)
            return value

        c["mlp"] = mlp
    if driver == "wm_encoder_mlp":

        def encoder_mlp(state, target, context, event):
            return model.net(
                model.inputs(model.features(state, target, context, event))
            )

        joint = compile_fn(encoder_mlp)
        return lambda state, target, context, event: c["integrate"](
            state, joint(state, target, context, event), context
        )
    if driver == "wm_mlp_integrate":

        def mlp_integrate(state, encoded, context):
            return c["integrate"](state, model.net(encoded), context)

        joint = compile_fn(mlp_integrate)
        return lambda state, target, context, event: joint(
            state, c["encoder"](state, target, context, event), context
        )

    def predict(state, target, context, event):
        encoded = (
            c["encoder"](state, target, context, event)
            if driver == "wm_encoder"
            else c["normalization"](c["features"](state, target, context, event))
        )
        return c["integrate"](state, c["mlp"](encoded), context)

    return predict


def update_drift(reference, actual, named_parameters):
    """Locate update differences and corresponding applied gradients; no fixes."""
    rows = []
    for index, (name, _) in enumerate(named_parameters):
        key = f"parameter/{name}"
        a, b = reference[key], actual[key]
        error = (a - b).abs()
        flat_index = int(error.reshape(-1).argmax())
        if float(error.reshape(-1)[flat_index]) == 0:
            continue
        coordinate, remaining = [], flat_index
        for width in reversed(a.shape):
            coordinate.append(remaining % width)
            remaining //= width
        rows.append(
            {
                "parameter": name,
                "flat_index": flat_index,
                "coordinate": list(reversed(coordinate)),
                "shape": list(a.shape),
                "max_abs_update_difference": float(error.reshape(-1)[flat_index]),
                "reference_parameter_after_adam": float(a.reshape(-1)[flat_index]),
                "candidate_parameter_after_adam": float(b.reshape(-1)[flat_index]),
                "reference_applied_gradient": float(
                    reference[f"gradient/{index}"].reshape(-1)[flat_index]
                ),
                "candidate_applied_gradient": float(
                    actual[f"gradient/{index}"].reshape(-1)[flat_index]
                ),
            }
        )
    return rows
