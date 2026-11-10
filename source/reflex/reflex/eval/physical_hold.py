"""Evaluation only: shared pose-agnostic hold and per-throw control diagnostics.

A hand/forearm/elbow contact is required; torso-only support never qualifies.
Thresholds are fixed across S3/A/B/C, independent of reward configuration.
"""
import torch


class PhysicalHoldState:
    def __init__(self,n,device):
        self.age=torch.zeros(n,device=device)
        self.recent=torch.full_like(self.age,100.)
        self.contact=torch.zeros(n,dtype=torch.bool,device=device)
        self.ground=self.contact.clone();self.escape=self.contact.clone();self.success=self.contact.clone()
        self.control=self.age.clone();self.continuous=self.age.clone();self.longest=self.age.clone()

    def reset(self,ids):
        for v in (self.age,self.contact,self.ground,self.escape,self.success,self.control,self.continuous,self.longest):v[ids]=0
        self.recent[ids]=100.

    def update(self,*,dt,active,safe,arm_contact,forbidden,ground_contact,bottom,radius,relative_speed,angular_speed):
        hit=active&safe&arm_contact&~ground_contact&(bottom>.08)
        self.contact|=hit
        self.recent=torch.where(hit,0.,self.recent+dt)
        self.ground|=active&(ground_contact|(bottom<=.08))
        self.escape|=active&self.contact&(radius>1.2)
        controlled=active&safe&self.contact&~self.ground&~self.escape&~forbidden&(
            self.recent<=.12+1e-6)&(radius<.75)&(bottom>.08)
        stable=controlled&(relative_speed<.5)&(angular_speed<2.)
        self.age=torch.where(stable,self.age+dt,0.)
        self.success|=self.age>=.6-1e-6
        self.control+=controlled.float()*dt
        self.continuous=torch.where(controlled,self.continuous+dt,0.)
        self.longest=torch.maximum(self.longest,self.continuous)

    def row(self,i):
        return dict(physical_hold_success=bool(self.success[i]),physical_contact=bool(self.contact[i]),
            physical_ground=bool(self.ground[i]),physical_escape=bool(self.escape[i]),
            control_seconds=float(self.control[i]),longest_control_seconds=float(self.longest[i]))


class PhysicalHoldEvaluator:
    def __init__(self,env):
        self.env=env;self.state=PhysicalHoldState(env.num_envs,env.device)
        self.prev=None;self.index=torch.zeros(env.num_envs,dtype=torch.long,device=env.device)
        self.rows={};self.observed=torch.zeros_like(self.index,dtype=torch.bool)
        self.arm=None
        self.diagnostics=[{} for _ in range(env.num_envs)]

    def update(self):
        from reflex.envs.escape_penalty import torso_relative_kinematics
        from isaaclab.utils.math import quat_apply_inverse
        env=self.env;sensor=env.scene['buffer_box_contacts'];_=sensor.data
        if self.arm is None:
            names=sensor.robot_filter_names
            self.arm=[j for j,n in enumerate(names) if any(k in n for k in ('hand','wrist','elbow','forearm'))]
            self.bad=[j for j,n in enumerate(names) if any(k in n for k in ('head','shoulder'))]
            if not self.arm:raise RuntimeError('physical hold requires explicit arm contact filters')
        launch=env._box_launched
        if self.prev is None:self.prev=launch.clone()
        rising=launch&~self.prev
        for j in rising.nonzero().flatten().tolist():
            if self.index[j]>0:self.rows[(j,int(self.index[j]))]=dict(self.state.row(j),**self.diagnostics[j])
        for j in rising.nonzero().flatten().tolist():self.diagnostics[j]={}
        self.index+=rising.long();self.state.reset(rising);self.prev=launch.clone()
        self.observed|=rising
        robot=env.scene['robot'];d=robot.data;box=env.scene['balloon'].data
        i=list(robot.body_names).index('torso_link')
        q=d.body_quat_w[:,i];w,x,y,z=q.unbind(-1)
        safe=(d.root_pos_w[:,2]-env.scene.env_origins[:,2]>.45)&(
            d.body_pos_w[:,i,2]-env.scene.env_origins[:,2]>.55)&(1-2*(x*x+y*y)>.25)
        bw,bx,by,bz=box.root_quat_w.unbind(-1)
        row=torch.stack((2*(bx*bz-bw*by),2*(by*bz+bw*bx),1-2*(bx*bx+by*by)),-1)
        bottom=box.root_pos_w[:,2]-env.scene.env_origins[:,2]-(row.abs()*env._box_half_extents).sum(-1)
        f=sensor.pair_force_history.norm(dim=-1)
        arm=(f[...,self.arm]>1.).flatten(1).any(-1)
        forbidden=(f[...,self.bad]>1.).flatten(1).any(-1) if self.bad else torch.zeros_like(arm)
        floor=(f[...,sensor.robot_filter_count:]>1.).flatten(1).any(-1)
        pos,vel=torso_relative_kinematics(env)
        omega=quat_apply_inverse(q,box.root_ang_vel_w-d.body_ang_vel_w[:,i]).norm(dim=-1)
        hold=getattr(getattr(env,'_buffer_state',None),'hold_init',torch.zeros_like(launch))
        previous_contact=self.state.contact.clone()
        self.state.update(dt=env.step_dt,active=launch&self.observed&~hold,safe=safe,arm_contact=arm,
            forbidden=forbidden,ground_contact=floor,bottom=bottom,radius=pos.norm(dim=-1),
            relative_speed=vel.norm(dim=-1),angular_speed=omega)

        # Evaluation diagnostics have no path back into rewards/actions.
        active=launch&self.observed&~hold
        left=[j for j in self.arm if 'left' in sensor.robot_filter_names[j]]
        right=[j for j in self.arm if 'right' in sensor.robot_filter_names[j]]
        bilateral=((f[...,left]>1.).flatten(1).any(-1)&(f[...,right]>1.).flatten(1).any(-1)) if left and right else torch.zeros_like(arm)
        pitch=torch.atan2(2*(w*y-x*z),1-2*(x*x+y*y))
        knees=[j for j,n in enumerate(robot.joint_names) if 'knee' in n]
        for j in active.nonzero().flatten().tolist():
            diag=self.diagnostics[j]
            for key,value in (('arm_contact_seconds',arm[j]),('bilateral_arm_contact_seconds',bilateral[j]),
                              ('head_shoulder_contact_seconds',forbidden[j])):
                diag[key]=diag.get(key,0.)+float(value)*env.step_dt
            diag['max_back_lean_degrees']=max(diag.get('max_back_lean_degrees',0.),float((-pitch[j]).clamp_min(0)*180/torch.pi))
            for key in ('shoulder','elbow'):
                ids=[k for k,n in enumerate(robot.joint_names) if key in n]
                angles=d.joint_pos[j,ids].tolist()
                lo=diag.setdefault(key+'_min',angles.copy());hi=diag.setdefault(key+'_max',angles.copy())
                diag[key+'_min']=[min(a,b) for a,b in zip(lo,angles)]
                diag[key+'_max']=[max(a,b) for a,b in zip(hi,angles)]
            if bool(self.state.contact[j]&~previous_contact[j]):
                diag['first_contact_pitch']=float(pitch[j])
                diag['first_contact_torso_omega']=d.body_ang_vel_w[j,i].tolist()
                diag['first_contact_pelvis_velocity']=d.root_lin_vel_w[j].tolist()
                diag['first_contact_knees']=d.joint_pos[j,knees].tolist()

    def finish(self,throw_log):
        for j in range(self.env.num_envs):
            if self.index[j]>0:self.rows[(j,int(self.index[j]))]=dict(self.state.row(j),**self.diagnostics[j])
        for row in throw_log:
            key=(row['env'],row['idx'])
            if key not in self.rows:raise RuntimeError(f'missing physical-hold record {key}')
            row.update(self.rows[key])
