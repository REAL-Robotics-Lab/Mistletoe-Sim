# modified rsl-rl play file to publish to the COMMAND LCM channel for manual command testing

# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch

from rsl_rl.runners import OnPolicyRunner

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)

import csv
import time
import pandas as pd
import numpy as np
import torch.nn as nn
import lcm
from exlcm import quad_command_t
import time 

def write_obs_to_csv(file_name, headers):
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # Write headers once at the beginning

def append_obs_to_csv(file_name, obs):
    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file)
        # for row in obs.cpu().numpy():
        #     writer.writerow(row)
        # print(obs.cpu().numpy()[0])
    # Count the number of rows in the CSV file
    with open(file_name, mode='r') as file:
        reader = csv.reader(file)
        num_rows = sum(1 for row in reader)
    
    # Print the number of rows in the CSV file
    print(f"Number of rows in the CSV file: {num_rows}")
    
# Example usage
file_name = 'dataset.csv'
headers = [
    'base_lin_vel_x', 'base_lin_vel_y', 'base_lin_vel_z',
    'base_ang_vel_x', 'base_ang_vel_y', 'base_ang_vel_z',
    'projected_gravity_x', 'projected_gravity_y', 'projected_gravity_z',
    'velocity_command_x', 'velocity_command_y', 'velocity_command_z',
    'joint_pos_1', 'joint_pos_2', 'joint_pos_3', 'joint_pos_4', 'joint_pos_5', 'joint_pos_6',
    'joint_pos_7', 'joint_pos_8', 'joint_pos_9', 'joint_pos_10', 'joint_pos_11', 'joint_pos_12',
    'joint_vel_1', 'joint_vel_2', 'joint_vel_3', 'joint_vel_4', 'joint_vel_5', 'joint_vel_6',
    'joint_vel_7', 'joint_vel_8', 'joint_vel_9', 'joint_vel_10', 'joint_vel_11', 'joint_vel_12',
    'joint_action_1', 'joint_action_2', 'joint_action_3', 'joint_action_4', 'joint_action_5', 'joint_action_6',
    'joint_action_7', 'joint_action_8', 'joint_action_9', 'joint_action_10', 'joint_action_11', 'joint_action_12'
]

write_obs_to_csv(file_name, headers)

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.layer1 = nn.Linear(45, 256)
        self.layer2 = nn.Linear(256, 128)
        self.output_layer = nn.Linear(128, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.output_layer(x)
        return x

model = SimpleMLP().to('cuda')
model_path = 'source/standalone/workflows/rsl_rl/simple_mlp_model.pth'

# Load the model state from the file for inference
model.load_state_dict(torch.load(model_path))
model.eval()
print(f"Model loaded from {model_path}")

def parse_RL_inference_output(inference_output):
    # for whatever reason isaaclab scales output by 0.25, and then convert to radians, then convert to format that can be used by us
    converted_output = ((inference_output[0] * 0.25)/(2*np.pi)).tolist()
    sorted_output = []
    for i in range(4):
        for j in range(12):
            if j%4==i:
                sorted_output.append(converted_output[j])

    return sorted_output

def main():
    lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=1")    
    msg = quad_command_t()
    msg.manual_command = True

    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(
        ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    )
    export_policy_as_onnx(ppo_runner.alg.actor_critic, path=export_model_dir, filename="policy.onnx")

    # reset environment
    obs, _ = env.get_observations()

    time_steps = 0

    # simulate environment
    max = 0
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            print(parse_RL_inference_output(actions))

            # make sure to convert to revs that is used for the robot.
            msg.position = (actions[0])
            # print(actions[0][0].item())
            msg.timestamp = time.time_ns()
            msg.manual_command = True
            # env stepping
            obs, _, _, _ = env.step(actions)
            lc.publish("COMMAND", msg.encode())
            # time_steps += 1 
            # print(time_steps)


    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()

    df = pd.read_csv(file_name)
    print(f'Number of datapoints collected: {len(df)}')

    # close sim app
    simulation_app.close()