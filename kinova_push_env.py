# Gen3 Push RL Environment
# Train a policy to push a sphere to a target location
# Framework: NVIDIA Isaac Lab 2.1.0
# Author: Youcef Sellai

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab_assets.robots.kinova import KINOVA_GEN3_N7_CFG
from isaaclab.assets import AssetBaseCfg

@configclass
class KinovaPushSceneCfg(InteractiveSceneCfg):
    # robot defined here with ENV_REGEX_NS
    robot: ArticulationCfg = KINOVA_GEN3_N7_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )
    # sphere defined here
    sphere: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Sphere",
        spawn=sim_utils.SphereCfg(
            radius=0.05,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                linear_damping=0.5,
                angular_damping=0.5,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0)
            )
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.45, 0.0, 0.15)
        )
    )
    # ground plane
    ground: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg()
        )

@configclass
class KinovaPushEnvCfg(DirectRLEnvCfg):
    sim: SimulationCfg = SimulationCfg(dt=0.01, render_interval=2)
    scene: KinovaPushSceneCfg = KinovaPushSceneCfg(
        num_envs=512,
        env_spacing=2.5
    )
    episode_length_s: float = 5.0
    decimation: int = 2
    num_observations: int = 16
    num_actions: int = 3
    observation_space: int = 16
    action_space: int = 3

# environment class

class KinovaPushEnv(DirectRLEnv):

    cfg: KinovaPushEnvCfg

    def __init__(self, cfg: KinovaPushEnvCfg, render_mode: str = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
                
        self.step_count = 0
        
        # IK controller stup
        ik_cfg = DifferentialIKControllerCfg(
            command_type = "pose",
            use_relative_mode = False,
            ik_method = "dls",

        )
        self.ik_controller = DifferentialIKController(
            cfg = ik_cfg,
            num_envs = self.num_envs,
            device = self.device

        )

        # robot entity cinfig for IK
        self.robot_entity_cfg = SceneEntityCfg(
            "robot",
            joint_names = ["joint_[1-7]"],
            body_names = ["end_effector_link"]
        )
        self.robot_entity_cfg.resolve(self.scene)
        self.ee_jacobi_idx = self.robot_entity_cfg.body_ids[0] - 1

        # target position for the sphere to reach

        self.target_pos = torch.tensor([0.5, 0.0, 0.15], device = self.device)

        print(f"Environment initialized with {self.num_envs} envs")

    # telling isaac lab what object exist in the scene

    def _setup_scene(self):
        self.robot = self.scene.articulations["robot"]
        self.sphere = self.scene.rigid_objects["sphere"]
        
        # clone environments
        self.scene.clone_environments(copy_from_source=False)
        # ground plane is handled by KinovaPushSceneCfg

    def _pre_physics_step(self, actions: torch.Tensor):
        # store actions for use in _apply_action
        self.action_buf = actions.clone()
       

    def _apply_action(self):


        # actions shpae: (512, 3)  - one velocity command per joint per environment
        actions = self.action_buf  # Isaac Lab stores actions here automatically
        self.actions = actions.clone()

        # get current end effector for all environments
        ee_pos_w = self.robot.data.body_pos_w[:, self.ee_jacobi_idx, :]
        ee_rot_w = self.robot.data.body_quat_w[:, self.ee_jacobi_idx, :]

        # convert to base frame

        root_pos_w = self.robot.data.root_pos_w
        root_quat_w = self.robot.data.root_quat_w
        ee_pos_b, ee_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_rot_w)

        # get jacobian
        jacobian = self.robot.root_physx_view.get_jacobians()[ :, self.ee_jacobi_idx, :, self.robot_entity_cfg.joint_ids]

        # get current joint positions
        joint_pos = self.robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids]

        # convert policy actions to a  target position

        ee_target = ee_pos_b + actions[:, :3]* 0.02 #reduced so movement become smaller
        ee_rot_target = ee_quat_b

        # combining position and rotation into one tensor
        # because IK controller expects position and rotation as one tensor of 7 values

        target_pose = torch.cat([ee_target, ee_rot_target], dim=1)
        
        # telling IK where to go
        
        self.ik_controller.set_command(target_pose)

        # compute IK
        # IK solver takes current state and target, outputs joint position targets

        joint_targets = self.ik_controller.compute( ee_pos_b, ee_quat_b, jacobian, joint_pos)

        # apply to robot

        self.robot.set_joint_position_target(joint_targets, joint_ids = self.robot_entity_cfg.joint_ids)
        self.robot.write_data_to_sim()
    
    # the _get_observations() method is what the roobot sees, every step isaacsim calls this method and expects a tensor of shape (512,13) because 13 numbers are describing the current state of each 512 envs
    # it looks at current 7 joints, current end effector in 3d space, sphere in 3d space

    def _get_observations(self) -> dict:

        #joint pos
        joint_pos = self.robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids]
        root_pos = self.robot.data.root_pos_w

        # end effector position in 3d local frame
        ee_pos_w = self.robot.data.body_pos_w[:, self.robot_entity_cfg.body_ids[0], :]
        ee_pos = ee_pos_w - root_pos  # local
        
        #sphere position in 3d local frame
        sphere_pos_w = self.sphere.data.root_pos_w
        sphere_pos = sphere_pos_w - root_pos  # local

        # target position, expand to match batch size (512, 3)

        target = self.target_pos.unsqueeze(0).expand(self.num_envs,-1)

        #concatenate all into 1 tensor shape (512, 16)
        # isaaclab expects the key "policy"
        
        obs = torch.cat([joint_pos, ee_pos, sphere_pos, target], dim = -1)

        return {"policy": obs}
    
    # the reward method, most important method
    # main objective : getting the sphere close to the target
    # gettintg the ee close to the sphere to help the robot learn to approach first
    # bonus for reaching the target
    # mall penalty every step to encourage the robot to not waste time

    def _get_rewards(self) -> torch.Tensor:

        #sphere pos
        sphere_pos = self.sphere.data.root_pos_w

        # ee pos
        ee_pos = self.robot.data.body_pos_w[:, self.robot_entity_cfg.body_ids[0], :]

        # root position
        root_pos = self.robot.data.root_pos_w

        # MUST convert to LOCAL cordinates

        sphere_local = sphere_pos - root_pos
        ee_local = ee_pos - root_pos
        target_local = self.target_pos 

        # distance from sphere to target

        dist_sphere_to_target = torch.norm(sphere_local - target_local, dim =-1)

        # REWARDS
        # REWARD MAIN : sphere getting close to target

        reward_sphere = 5.0 * (1.0 - torch.tanh(dist_sphere_to_target))
        # REWARD APPROACHING : ee getting close to sphere
        
        #reward_approach could be an issue i leave it in comment and take it out of the total reward
        #reward_approach = 0.05 * (1.0 - torch.tanh(dist_ee_to_sphere)) # changing from 0.3 to 0.1 because policy found way to optimize by just approaching sphere

        # takeout reward_x_align cause it rewards for staying still, maybe is the cause for minimal movements
        #reward_x_align = 0.3 * (1.0 - torch.tanh(ee_sphere_x_diff))


        # BONUS : sphere reached the target

        success_bonus = torch.where(

            dist_sphere_to_target < 0.1,
            torch.ones_like(dist_sphere_to_target)*100.0,  # increasing from 10 to 50 to make sucess more attractive
            torch.zeros_like(dist_sphere_to_target)

        )
    
        #small penalty each step to encourage efficiency ( return a tensor always), lowered to 0.005 to put less pressure to end fast

        time_penalty = torch.full((self.num_envs,), -0.005, device=self.device)

        # total reward
        reward = reward_sphere + success_bonus + time_penalty 

        return reward
    
    # when to end an episode

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:


        #episode time out if too many steps passed, isaaclab track this automaticly

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # episode succeeds if sphere reaches target
        sphere_pos = self.sphere.data.root_pos_w
        root_pos = self.robot.data.root_pos_w
        sphere_local = sphere_pos - root_pos
        dist_sphere_to_target = torch.norm(sphere_local - self.target_pos, dim=-1)
        success = dist_sphere_to_target < 0.1

        # tracking sucess rate and positions to know ho training goes

        self.step_count += 1
        if self.step_count % 1000 == 0:

            sphere_pos_w = self.sphere.data.root_pos_w
            ee_pos_w = self.robot.data.body_pos_w[:, self.robot_entity_cfg.body_ids[0], :]
            root_pos_w = self.robot.data.root_pos_w

            sphere_local = sphere_pos_w - root_pos_w
            ee_local = ee_pos_w - root_pos_w
            
            avg_sphere_x = sphere_local[:, 0].mean().item()
            avg_ee_x = ee_local[:, 0].mean().item()
            dist_ee_to_sphere = torch.norm(ee_local - sphere_local, dim=-1).mean().item()
            success_rate = (dist_sphere_to_target < 0.1).float().mean().item()
            
            print(f"Step {self.step_count} | Sphere_X: {avg_sphere_x:.3f} | EE_X: {avg_ee_x:.3f} | EE-Sphere dist: {dist_ee_to_sphere:.3f} | Success: {success_rate:.3f}")

        #isaaclab expects 2 tensor: (terminated, timed_out)
        #terminated = episode ended because success or failure
        # timed_out = episode ended because time ran out
        return success, time_out
    
    # reset specific environments
    # when episode ends (success or timeout) isaaclab calls this to reset those environments

    def _reset_idx(self, env_ids:torch.Tensor):
        super()._reset_idx(env_ids)

        #reset robot to default pose
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids = env_ids)
        self.robot.reset(env_ids)

        # reset sphere to RANDOM position in front of robot
        sphere_pos = torch.zeros(len(env_ids), 3, device=self.device)
        sphere_pos[:, 0] = torch.FloatTensor(len(env_ids)).uniform_(0.3, 0.6).to(self.device)
        sphere_pos[:, 1] = torch.FloatTensor(len(env_ids)).uniform_(-0.2, 0.2).to(self.device)
        sphere_pos[:, 2] = 0.15

        # to convert local to world
        env_origins = self.scene.env_origins[env_ids]
        sphere_pos_world = sphere_pos + env_origins

        # quaternion (no rotation)
        sphere_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).repeat(len(env_ids), 1)

        # velocity (zeros)
        sphere_vel = torch.zeros(len(env_ids), 6, device=self.device)

        # write_root_state_to_sim expects: pos(3) + quat(4) + vel(6) = 13 values
        self.sphere.write_root_state_to_sim(
            torch.cat([sphere_pos_world, sphere_quat, sphere_vel], dim=1),
            env_ids=env_ids
        )

        #reset IK controller for these environments
        self.ik_controller.reset(env_ids)
