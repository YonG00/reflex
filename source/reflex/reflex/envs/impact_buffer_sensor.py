"""Keep FILTERED pair forces at physics frequency, including brief rebounds."""
import torch
from isaaclab.sensors import ContactSensor


class ImpactPairContactSensor(ContactSensor):
    """One box reporting body, multiple explicit arm-body filters.

    Some Isaac Lab releases keep history only for unfiltered net forces. Keep
    our own pair history so a hit in any decimation substep is retained. A
    positive history_length makes SensorBase.update eagerly update each physics
    tick, even when scene.lazy_sensor_update is enabled.
    """
    def _update_buffers_impl(self, env_ids):
        env_ids = self._ALL_INDICES if env_ids is None else env_ids
        super()._update_buffers_impl(env_ids)
        forces = self._data.force_matrix_w
        if forces is None:
            raise RuntimeError('Impact sensor requires filtered force_matrix_w')
        if not hasattr(self, 'pair_force_history'):
            self.pair_force_history = torch.zeros(
                (forces.shape[0], self.cfg.history_length, *forces.shape[1:]),
                device=forces.device, dtype=forces.dtype)
        history = self.pair_force_history[env_ids].clone()
        history[:, 1:] = history[:, :-1].clone()
        history[:, 0] = forces[env_ids]
        self.pair_force_history[env_ids] = history

    def clear_history(self, env_ids):
        if hasattr(self, 'pair_force_history'):
            self.pair_force_history[env_ids] = 0

    def reset(self, env_ids=None):
        super().reset(env_ids)
        self.clear_history(slice(None) if env_ids is None else env_ids)
