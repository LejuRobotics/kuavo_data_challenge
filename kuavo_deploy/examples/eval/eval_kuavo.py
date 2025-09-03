# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script demonstrates how to evaluate a pretrained policy from the HuggingFace Hub or from your local
training outputs directory. In the latter case, you might want to run kuavo_train/train_policy.py first.

It requires the installation of the 'gym_pusht' simulation environment. Install it by running:
```bash
pip install -e ".[pusht]"
```
"""

from lerobot_patches import custom_patches

from pathlib import Path

from sympy import im
from dataclasses import dataclass, field
import hydra
import gymnasium as gym
import imageio
import numpy
import torch
from tqdm import tqdm

from kuavo_train.wrapper.policy.diffusion.DiffusionPolicyWrapper import CustomDiffusionPolicyWrapper
from lerobot.utils.random_utils import set_seed
import datetime
import time
import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf
from torchvision.transforms.functional import to_tensor
from std_msgs.msg import Bool
import rospy
import threading

from configs.deploy.config_inference import load_inference_config
from kuavo_deploy.utils.logging_utils import setup_logger
log_model = setup_logger("model")
log_robot = setup_logger("robot")

def pause_callback(msg):
    if msg.data:
        pause_flag.set()
    else:
        pause_flag.clear()

def stop_callback(msg):
    if msg.data:
        stop_flag.set()

pause_sub = rospy.Subscriber('/kuavo/pause_state', Bool, pause_callback, queue_size=10)
stop_sub = rospy.Subscriber('/kuavo/stop_state', Bool, stop_callback, queue_size=10)
stop_flag = threading.Event()
pause_flag = threading.Event()

def check_control_signals():
    """检查控制信号"""
        # 添加调试信息
    log_model.info(f"🔍 检查控制信号 - 暂停状态: {pause_flag}, 停止状态: {stop_flag}")
    

def img_preprocess(image, device="cpu"):
    return to_tensor(image).unsqueeze(0).to(device, non_blocking=True)

def depth_preprocess(depth, device="cpu"):
    return torch.tensor(depth,dtype=torch.float32).unsqueeze(0).to(device, non_blocking=True)

def setup_policy(pretrained_path, policy_type, device=torch.device("cuda")):
    """
    Set up and load the policy model.
    
    Args:
        pretrained_path: Path to the checkpoint
        policy_type: Type of policy ('diffusion' or 'act')
        
    Returns:
        Loaded policy model and device
    """
    
    if device.type == 'cpu':
        log_model.warning("Warning: Using CPU for inference, this may be slow.")
        time.sleep(3)  
    
    if policy_type == 'diffusion':
        policy = CustomDiffusionPolicyWrapper.from_pretrained(Path(pretrained_path),strict=True)
    elif policy_type == 'act':
        raise ValueError(f"Unsupported policy type: {policy_type}")
    else:
        raise ValueError(f"Unsupported policy type: {policy_type}")
    
    policy.eval()
    policy.to(device)
    policy.reset()
    # Log model info
    log_model.info(f"Model loaded from {pretrained_path}")
    log_model.info(f"Model n_obs_steps: {policy.config.n_obs_steps}")
    log_model.info(f"Model device: {device}")
    
    return policy

def main(config_path: str, env: gym.Env):
    # load config
    cfg = load_inference_config(config_path)

    use_delta = cfg.use_delta
    eval_episodes = cfg.eval_episodes
    seed = cfg.seed
    start_seed = cfg.start_seed
    policy_type = cfg.policy_type
    task = cfg.task
    method = cfg.method
    timestamp = cfg.timestamp
    epoch = cfg.epoch
    env_name = cfg.env_name

    pretrained_path = Path(f"outputs/train/{task}/{method}/{timestamp}/epoch{epoch}")
    output_directory = Path(f"outputs/eval/{task}/{method}/{timestamp}/epoch{epoch}")
    # Create a directory to store the video of the evaluation
    output_directory.mkdir(parents=True, exist_ok=True)

    # set seed
    set_seed(seed=seed)

    # Select your device
    device = torch.device(cfg.device)

    policy = setup_policy(pretrained_path, policy_type, device)

    # Initialize evaluation environment to render two observation types:
    # an image of the scene and state/position of the agent.
    max_episode_steps = cfg.max_episode_steps
    env = env

    # We can verify that the shapes of the features expected by the policy match the ones from the observations
    # produced by the environment
    log_model.info(f"policy.config.input_features: {policy.config.input_features}")
    log_robot.info(f"env.observation_space: {env.observation_space}")

    # Similarly, we can check that the actions produced by the policy will match the actions expected by the
    # environment
    log_model.info(f"policy.config.output_features: {policy.config.output_features}")
    log_robot.info(f"env.action_space: {env.action_space}")

    # Log evaluation results
    log_file_path = output_directory / "evaluation.log"
    with log_file_path.open("w") as log_file:
        log_file.write(f"Evaluation Timestamp: {datetime.datetime.now()}\n")
        log_file.write(f"Total Episodes: {eval_episodes}\n")

    success_count = 0
    for episode in tqdm(range(eval_episodes), desc="Evaluating model", unit="episode"):
        # Reset the policy and environments to prepare for rollout
        policy.reset()
        numpy_observation, info = env.reset(seed=episode+start_seed)
        # print("numpy_observation",numpy_observation)

        # Prepare to collect every rewards and all the frames of the episode,
        # from initial state to final state.
        rewards = []
        # Render frame of the initial state
        # frames.append(env.render())
        cam_keys = [k for k in numpy_observation.keys() if "images" in k or "depth" in k]
        frame_map = {k: [] for k in cam_keys}

        step = 0
        done = False
        with tqdm(total=max_episode_steps, desc=f"Episode {episode+1}", unit="step", leave=False) as pbar:
            while not done:
                # --- Pause support: block here if pause_flag is set ---
                while pause_flag.is_set() and not stop_flag.is_set():
                    log_model.info("Paused. Waiting for resume signal...")
                    time.sleep(0.5)
                if stop_flag.is_set():
                    log_model.info("Stop flag detected during pause. Exiting loop.")
                    return
                
                start_time = time.time()
                # Prepare observation for the policy running in Pytorch

                observation = {}
                
                for k,v in numpy_observation.items():
                    if "images" in k:
                        observation[k] = img_preprocess(v, device=device)
                    elif "state" in k:
                        observation[k] = torch.from_numpy(v).float().unsqueeze(0).to(device, non_blocking=True)
                    elif "depth" in k:
                        observation[k] = depth_preprocess(v, device=device)

                with torch.inference_mode():
                    action = policy.select_action(observation)
                numpy_action = action.squeeze(0).cpu().numpy()
                log_model.debug(f"numpy_action: {numpy_action}")

                # Clip the action to the action space limits
                if use_delta:
                    if env.real:
                        if env.which_arm == "both":
                            for i in [(0, 7), (8, 15)]:
                                numpy_action[i[0]:i[1]] = np.clip(numpy_action[i[0]:i[1]], -0.05, 0.05) + env.start_state[i[0]:i[1]]
                            env.start_state = np.concatenate([
                                np.clip(numpy_action[0:7], env.action_space.low[0:7], env.action_space.high[0:7]),
                                np.clip(numpy_action[8:15], env.action_space.low[8:15], env.action_space.high[8:15])
                            ])
                        else:
                            numpy_action[:7] = np.clip(numpy_action[:7], -0.05, 0.05) + env.start_state[:7]
                            env.start_state = np.clip(numpy_action[:7], env.action_space.low[:7], env.action_space.high[:7])
                    else:
                        if env.which_arm == "both":
                            numpy_action[:7] = np.clip(numpy_action[:7], -0.1, 0.1)
                            numpy_action[8:15] = np.clip(numpy_action[8:15], -0.1, 0.1)
                            numpy_action[7] = np.clip(numpy_action[7], -0.05, 0.05)
                            numpy_action[15] = np.clip(numpy_action[15], -0.05, 0.05)
                            numpy_action += numpy_observation["observation.state"]

                # 执行动作
                numpy_observation, reward, terminated, truncated, info = env.step(numpy_action)
                rewards.append(reward)

                # 相机帧记录

                for k in cam_keys:
                    frame_map[k].append(observation[k].squeeze(0).cpu().numpy().transpose(1, 2, 0))

                # The rollout is considered done when the success state is reached (i.e. terminated is True),
                # or the maximum number of iterations is reached (i.e. truncated is True)
                done = terminated | truncated | done
                step += 1

                end_time = time.time()
                log_model.info(f"Step {step} time: {end_time - start_time:.3f}s")
                
                # Update progress bar
                status = "Success" if terminated else "Running"
                pbar.set_postfix({
                    "Reward": f"{reward:.3f}",
                    "Status": status,
                    "Total Reward": f"{sum(rewards):.3f}"
                })
                pbar.update(1)

        if terminated:
            success_count += 1
            log_model.info(f"✅ Episode {episode+1}: Success! Total reward: {sum(rewards):.3f}")
        else:
            log_model.info(f"❌ Episode {episode+1}: Failed! Total reward: {sum(rewards):.3f}")

        # Get the speed of environment (i.e. its number of frames per second).
        fps = env.ros_rate
        
        # Encode all frames into a mp4 video.
        for cam in cam_keys:
            frames = frame_map[cam]
            output_path = output_directory / f"rollout_{episode}_{cam}.mp4"
            imageio.mimsave(str(output_path), frames, fps=fps)

        # print(f"Video of the evaluation is available in '{video_path}'.")

        with log_file_path.open("a") as log_file:
            log_file.write("\n")
            log_file.write(f"Rewards per Episode: {numpy.array(rewards).sum()}")

    with log_file_path.open("a") as log_file:
        log_file.write("\n")
        log_file.write(f"Success Count: {success_count}\n")
        log_file.write(f"Success Rate: {success_count / eval_episodes:.2f}\n")

    # Display final statistics
    log_model.info("\n" + "="*50)
    log_model.info(f"🎯 Evaluation completed!")
    log_model.info(f"📊 Success count: {success_count}/{eval_episodes}")
    log_model.info(f"📈 Success rate: {success_count / eval_episodes:.2%}")
    print(f"📁 Videos and logs saved to: {output_directory}")
    print("="*50)

def kuavo_eval(config_path: Path, env: gym.Env):
    main(config_path, env)

if __name__ == "__main__":
    config_path = Path("test.yaml")
    env = gym.make(
        "Kuavo-Real",
        max_episode_steps=150,
        config_path=config_path,
    )