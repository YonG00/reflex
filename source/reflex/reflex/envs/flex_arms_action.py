"""C-only action adapter; preserve original two-stage/gimbal and elbow behavior."""
import torch
from reflex.envs.gimbal_action import GimbalJointPositionAction
from reflex.envs.flex_arms import ShoulderRamp,phase_mask


class FlexArmsAction(GimbalJointPositionAction):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        ids=list(range(len(self._asset.joint_names))) if isinstance(self._joint_ids,slice) else list(self._joint_ids)
        self._flex_cols=[c for c,j in enumerate(ids) if 'shoulder' in self._asset.joint_names[j]]
        self._flex_joints=[ids[c] for c in self._flex_cols]
        if not isinstance(self._scale,torch.Tensor) or self._scale.ndim!=2:
            raise ValueError('C requires the original per-joint S3 scale tensor')
        if not torch.allclose(self._scale[:,self._flex_cols],torch.full_like(self._scale[:,self._flex_cols],.02)):
            raise ValueError('C expects shoulder scale .02 before ramp')
        elbow=[c for c,j in enumerate(ids) if 'elbow' in self._asset.joint_names[j]]
        if not torch.allclose(self._scale[:,elbow],torch.full_like(self._scale[:,elbow],.35)):
            raise ValueError('C must retain original elbow scale .35')
        self._ramp=ShoulderRamp(env.num_envs,env.device)

    def process_actions(self,actions):
        active=phase_mask(self._env)
        scale=self._ramp.update(int(self._env.common_step_counter),self._env.step_dt,active)
        original=self._scale
        self._scale=original.clone();self._scale[:,self._flex_cols]=scale[:,None]
        try:super().process_actions(actions)
        finally:self._scale=original
        # Keep waiting exactly on the original processing path. In receiving
        # windows retain the pre-existing physical joint limit guard as well.
        lim=getattr(self._asset.data,'joint_pos_limits',None)
        if lim is None:lim=self._asset.data.soft_joint_pos_limits
        current=self._processed_actions[:,self._flex_cols]
        clipped=current.clamp(lim[:,self._flex_joints,0],lim[:,self._flex_joints,1])
        self._processed_actions[:,self._flex_cols]=torch.where(active[:,None],clipped,current)
        targets=self._processed_actions[:,self._flex_cols]
        previous=getattr(self,'_previous_flex_targets',targets)
        self._env._flex_target_delta_max=(targets-previous).abs().amax()
        self._previous_flex_targets=targets.clone()
        for key in ('shoulder','elbow'):
            js=[j for j,n in enumerate(self._asset.joint_names) if key in n]
            if js and hasattr(self._asset.data,'joint_pos'):
                angles=self._asset.data.joint_pos[:,js]
                low=getattr(self._env,'_flex_'+key+'_low',angles)
                high=getattr(self._env,'_flex_'+key+'_high',angles)
                setattr(self._env,'_flex_'+key+'_low',torch.minimum(low,angles))
                setattr(self._env,'_flex_'+key+'_high',torch.maximum(high,angles))
        self._env._flex_scale=scale
        self._env._flex_target_clip=(current-clipped).abs().amax(-1)
