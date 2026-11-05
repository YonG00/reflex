import os

from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from isaaclab_rl.rsl_rl import export_policy_as_onnx

from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


class MyOnPolicyRunner(OnPolicyRunner):
    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if self.logger_type in ["wandb"]:
            import wandb  # only needed with --logger wandb

            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            export_policy_as_onnx(self.alg.policy, normalizer=self.obs_normalizer, path=policy_path, filename=filename)
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))


class MotionOnPolicyRunner(OnPolicyRunner):
    def __init__(
        self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu", registry_name: str = None
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name

    def load(self, path, load_optimizer=True, map_location=None):
        infos = super().load(path, load_optimizer=load_optimizer, map_location=map_location)
        if load_optimizer:
            self.alg.learning_rate = self.alg.optimizer.param_groups[0]["lr"]
        if load_optimizer and getattr(self.alg, "amp_ready", False):
            state = infos.get("reflex_amp_state") if isinstance(infos, dict) else None
            if state is None:
                print("[AMP checkpoint] LEGACY: no AMP state; discriminator/Adam, ramp and replay "
                      "are reinitialized. This is not a complete AMP resume.", flush=True)
            else:
                self.alg.load_amp_state_dict(state)
        return infos

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        if getattr(self.alg, "amp_ready", False):
            infos = dict(infos) if isinstance(infos, dict) else {"user_infos": infos}
            infos["reflex_amp_state"] = self.alg.amp_state_dict()
        super().save(path, infos)
        if self.logger_type in ["wandb"]:
            import wandb  # only needed with --logger wandb

            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            # rsl_rl version compat: obs_normalizer may live on runner OR alg.
            normalizer = getattr(self, "obs_normalizer", None)
            if normalizer is None:
                normalizer = getattr(self.alg, "obs_normalizer", None)
            try:
                export_motion_policy_as_onnx(
                    self.env.unwrapped, self.alg.policy, normalizer=normalizer,
                    path=policy_path, filename=filename,
                )
                attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
                wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
            except Exception as _e:
                print(f"[MotionOnPolicyRunner.save] WARN ONNX export skipped: {_e!r}", flush=True)

            # link the artifact registry to this run
            if self.registry_name is not None:
                wandb.run.use_artifact(self.registry_name)
                self.registry_name = None
