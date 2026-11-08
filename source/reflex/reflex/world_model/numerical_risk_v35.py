"""Descriptive numerical diagnostics, never replacement acceptance tolerances."""

import torch


def vector_summary(reference, actual):
    a, b = (
        reference.detach().cpu().float().reshape(-1),
        actual.detach().cpu().float().reshape(-1),
    )
    if a.shape != b.shape or not a.numel():
        raise ValueError("Nonempty matching tensors required")
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        return {"finite": False}
    difference = b - a
    na, nb = a.norm(), b.norm()
    direction_defined = bool((na > 0) & (nb > 0))
    return {
        "finite": True,
        "max_abs": float(difference.abs().max()),
        "rms": float(difference.square().mean().sqrt()),
        "reference_norm": float(na),
        "candidate_norm": float(nb),
        "relative_l2": float(difference.norm() / na) if na > 0 else None,
        "direction_defined": direction_defined,
        "cosine": float(torch.dot(a / na, b / nb).clamp(-1, 1))
        if direction_defined
        else None,
        "sign_disagreements": int(((a > 0) & (b < 0) | (a < 0) & (b > 0)).sum()),
        "elements": a.numel(),
    }


def physical_summary(reference, actual, weights=None):
    if reference.shape != actual.shape or reference.shape[-1] != 84:
        raise ValueError("Require matching 84-dimensional physical states")
    a, b = reference.detach().cpu().float(), actual.detach().cpu().float()
    groups = {
        "root_position_m": (0, 3),
        "joint_position_rad": (7, 36),
        "root_linear_velocity_m_s": (36, 39),
        "root_angular_velocity_rad_s": (39, 42),
        "joint_velocity_rad_s": (42, 71),
        "box_position_m": (71, 74),
        "box_linear_velocity_m_s": (78, 81),
        "box_angular_velocity_rad_s": (81, 84),
    }
    result = {
        name: vector_summary(a[..., start:end], b[..., start:end])
        for name, (start, end) in groups.items()
    }
    if weights is not None:
        weights = weights.detach().cpu().float()
        if weights.shape != a.shape[:-1] or not bool(
            torch.isfinite(weights).all() and (weights >= 0).all()
        ):
            raise ValueError("Finite nonnegative trajectory weights required")
        total = weights.sum()
        for name, (start, end) in groups.items():
            error = (b[..., start:end] - a[..., start:end]).square()
            result[name]["reference_survival_weighted_rms"] = (
                float(
                    (
                        error.mul(weights[..., None]).sum() / (total * (end - start))
                    ).sqrt()
                )
                if total > 0
                else None
            )
    for name, start in (
        ("root_orientation_error_rad", 3),
        ("box_orientation_error_rad", 74),
    ):
        qa, qb = a[..., start : start + 4], b[..., start : start + 4]
        na, nb = qa.norm(dim=-1, keepdim=True), qb.norm(dim=-1, keepdim=True)
        valid = bool(
            torch.isfinite(qa).all()
            and torch.isfinite(qb).all()
            and (na > 1e-8).all()
            and (nb > 1e-8).all()
        )
        if not valid:
            result[name] = {"valid_quaternions": False}
            continue
        qa, qb = qa / na, qb / nb
        qb = qb * torch.where((qa * qb).sum(-1, keepdim=True) < 0, -1.0, 1.0)
        # Stable near zero; identical quaternions and opposite signs give zero.
        angle = 4 * torch.atan2((qa - qb).norm(dim=-1), (qa + qb).norm(dim=-1))
        result[name] = {
            "valid_quaternions": True,
            "max": float(angle.max()),
            "rms": float(angle.square().mean().sqrt()),
        }
    return result


def actor_risk(reference, actual, initial, named_parameters):
    names = [name for name, _ in named_parameters]
    ref_grad = torch.cat(
        [reference[f"gradient/{i}"].reshape(-1) for i in range(len(names))]
    )
    new_grad = torch.cat(
        [actual[f"gradient/{i}"].reshape(-1) for i in range(len(names))]
    )
    # Compare update deltas, not post-update weights dominated by initialization.
    ref_delta = torch.cat(
        [
            (
                reference[f"parameter/{name}"].cpu() - initial[name].detach().cpu()
            ).reshape(-1)
            for name in names
        ]
    )
    new_delta = torch.cat(
        [
            (actual[f"parameter/{name}"].cpu() - initial[name].detach().cpu()).reshape(
                -1
            )
            for name in names
        ]
    )
    return {
        "applied_gradient": vector_summary(ref_grad, new_grad),
        "actual_adam_delta": vector_summary(ref_delta, new_delta),
        "note": "Descriptive only; strict numeric gate is unchanged. Single update does not establish training equivalence.",
    }


def trajectory_risk(reference, actual):
    result = {
        "finite_all": all(
            bool(torch.isfinite(value).all()) for value in actual.values()
        ),
        "raw_actor_gradient": vector_summary(
            reference["actor_gradient"], actual["actor_gradient"]
        ),
        "physical_state": physical_summary(
            reference["physical_states"],
            actual["physical_states"],
            weights=torch.cat(
                (
                    torch.ones_like(reference["done"][:1]),
                    torch.cumprod(1 - reference["done"], dim=0)[:-1],
                ),
                0,
            )
            * reference["valid"].float(),
        ),
    }
    for key in ("reward", "style", "done"):
        result[key] = vector_summary(reference[key], actual[key])
    for key in ("truncated", "valid", "predicted_launch_probability"):
        result[key + "_disagreements"] = int((reference[key] != actual[key]).sum())
    # The producer packs physical/post/targets/mask, each of width 84.
    result["event_write_mask_disagreements"] = int(
        (
            reference["event_trace"][..., 252:336]
            != actual["event_trace"][..., 252:336]
        ).sum()
    )
    survival_a = torch.cumprod(1 - reference["done"], dim=0)
    survival_b = torch.cumprod(1 - actual["done"], dim=0)
    result["survival_after_step"] = {
        str(h): vector_summary(survival_a[h - 1], survival_b[h - 1])
        for h in (1, 8, 16, 32)
        if h <= len(survival_a)
    }
    return result
