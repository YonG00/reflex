"""Reuse filtered substep force history, covering every actual robot body."""
from reflex.envs.impact_buffer_sensor import ImpactPairContactSensor


class EscapeRobotContactSensor(ImpactPairContactSensor):
    def _initialize_impl(self):
        from isaaclab import sim as sim_utils
        from pxr import UsdPhysics
        # Expand only the first environment, then use one filter per rigid body.
        # One wildcard filter summing opposing contacts could cancel to zero.
        robot_expr = self.cfg.filter_prim_paths_expr[0].rsplit('/', 1)[0]
        env_expr = self.cfg.prim_path.rsplit('/', 1)[0]
        first_env = sim_utils.find_matching_prims(env_expr)[0].GetPath().pathString
        robot_name = robot_expr.rsplit('/', 1)[1]
        prims = sim_utils.find_matching_prims(first_env + '/' + robot_name + '/.*')
        names = sorted(p.GetName() for p in prims if p.HasAPI(UsdPhysics.RigidBodyAPI))
        if not names:
            raise RuntimeError('escape_penalty: no robot rigid bodies found for contact filters')
        self.cfg.filter_prim_paths_expr = [robot_expr + '/' + name for name in names]
        print(f'[escape_penalty] box -> {len(names)} robot rigid-body filters: {names}', flush=True)
        super()._initialize_impl()
