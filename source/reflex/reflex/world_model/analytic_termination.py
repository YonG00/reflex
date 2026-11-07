"""SuperMimic-vibed soft termination, wxyz link-frame states (no learned head).

Formula reference: zita-ch/supermimic_vibed supermimic/mdp.py soft_termination,
commit 0525a23a2dcfa61f8543f98b84c075fe66d9d286. See V29 documentation.
"""

import torch


def rotation_zz(quaternion):
    # Unit quaternion, wxyz: the exact R[2,2] entry; no angle conversion.
    return 1 - 2 * (quaternion[..., 1].square() + quaternion[..., 2].square())


def soft_termination(height, quaternion, reference_height, reference_quaternion):
    height_error = (reference_height - height).abs()
    tilt_error = (rotation_zz(reference_quaternion) - rotation_zz(quaternion)).abs()
    p_height = torch.sigmoid((height_error - 0.25) / 0.02)
    p_tilt = torch.sigmoid((tilt_error - 0.8) / 0.04)
    return 1 - (1 - p_height) * (1 - p_tilt)


def probability(task, physical, next_phase):
    root = task.tracked_names.index("pelvis")
    return soft_termination(
        physical[..., 2],
        physical[..., 3:7],
        task.ref_p[next_phase, root, 2],
        task.ref_quat[next_phase, root],
    )
