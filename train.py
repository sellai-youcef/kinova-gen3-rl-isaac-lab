# Kinova Gen3 RL Training Script
# Author: Youcef Sellai

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Kinova Gen3 push policy")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# all imports after app launches
import torch
import skrl
from skrl.envs.wrappers.torch import wrap_env
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.trainers.torch import SequentialTrainer

from kinova_push_env import KinovaPushEnv, KinovaPushEnvCfg

# define simple neural network models for policy and value function
class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions=True)

        self.net = torch.nn.Sequential(
            torch.nn.Linear(16, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 3)
        )
        self.log_std = torch.nn.Parameter(torch.zeros(3))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), self.log_std.expand_as(
            self.net(inputs["states"])), {}

class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)
        
        self.net = torch.nn.Sequential(
            torch.nn.Linear(16, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 1)
        )

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}

def main():
    # create environment
    env_cfg = KinovaPushEnvCfg()
    env_cfg.scene.num_envs = 512
    env = KinovaPushEnv(cfg=env_cfg)
    
    # wrap for skrl
    env = wrap_env(env)
    
    device = "cuda:0"
    
    # create models
    models = {
        "policy": Policy(env.observation_space, env.action_space, device),
        "value": Value(env.observation_space, env.action_space, device)
    }
    
    # create memory
    memory = RandomMemory(memory_size=24, num_envs=512, device=device)
    
    # configure PPO
    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg["rollouts"] = 24
    cfg["learning_epochs"] = 8
    cfg["mini_batches"] = 4
    cfg["discount_factor"] = 0.99
    cfg["learning_rate"] = 3e-4
    cfg["experiment"]["write_interval"] = 100
    cfg["experiment"]["checkpoint_interval"] = 500
    cfg["experiment"]["directory"] = "logs/kinova_push"
    
    # create PPO agent
    agent = PPO(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device
    )
    
    # create trainer
    trainer_cfg = {"timesteps": 1000000, "headless": True}
    trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=agent)
    
    # start training
    print("Starting PPO training...")
    print(f"Training with 512 parallel environments")
    print(f"Running for 1000000 timesteps")
    trainer.train()
    
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()