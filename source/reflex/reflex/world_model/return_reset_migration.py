"""Opt-in addition of return-weighted reset to a full training resume."""


def validate_return_reset_migration(state, config, reinitialize_replay):
    if not reinitialize_replay:
        raise ValueError("Return-reset initialization requires fresh simulator/Replay")
    if state['config'].get('return_reset_max_step_reward') is not None or state.get('return_reset') is not None:
        raise ValueError("Existing return-reset statistics must be restored, not discarded")
    if config.return_reset_max_step_reward is None or config.return_reset_max_step_reward <= 0:
        raise ValueError("Return-reset migration requires an enabled positive reward scale")
    if state.get('actor_updates', 0) <= 0:
        raise ValueError("Use scratch initialization for untrained checkpoints")
    return {
        'from': None,
        'to': config.return_reset_max_step_reward,
        'sampler_statistics': 'fresh',
        'weights_optimizers_normalizers_preserved': True,
    }
