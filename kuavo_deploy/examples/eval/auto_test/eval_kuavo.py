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

import subprocess
import sys
from std_srvs.srv import Trigger, TriggerRequest, TriggerResponse
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
import json
from omegaconf import DictConfig, ListConfig, OmegaConf
from torchvision.transforms.functional import to_tensor
from std_msgs.msg import Bool
import rospy
import threading
import traceback
from geometry_msgs.msg import PoseStamped
from configs.deploy.config_inference import load_inference_config
from kuavo_deploy.utils.logging_utils import setup_logger
log_model = setup_logger("model")
log_robot = setup_logger("robot")

from kuavo_deploy.kuavo_env.kuavo_sim_env.KuavoSimEnv import KuavoSimEnv
from kuavo_deploy.kuavo_env.kuavo_real_env.KuavoRealEnv import KuavoRealEnv

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
    # 检查暂停状态
    while pause_flag.is_set():
        log_robot.info("🔄 机械臂运动已暂停")
        time.sleep(0.1)
        if stop_flag.is_set():
            log_robot.info("🛑 机械臂运动被停止")
            return False
    
    # 检查是否需要停止
    if stop_flag.is_set():
        log_robot.info("🛑 收到停止信号，退出机械臂运动")
        return False
        
    return True  # 正常继续
    

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

success_evt = threading.Event()
def env_success_callback(msg):
    log_model.info("env_success_callback!")
    if msg.data:
        success_evt.set()

# Globals to store latest tracked positions
latest_object_position = None
latest_marker1_position = None
latest_marker2_position = None

def box_grab_callback(msg):
    global latest_object_position
    p = msg.pose.position
    latest_object_position = [p.x, p.y, p.z]

def marker1_callback(msg):
    global latest_marker1_position
    p = msg.pose.position
    latest_marker1_position = [p.x, p.y, p.z]

def marker2_callback(msg):
    global latest_marker2_position
    p = msg.pose.position
    latest_marker2_position = [p.x, p.y, p.z]

def check_rostopics(task):
    topics = {}
    if "task1" in task:
        topics.update({
            "/mujoco/box_grab/pose": "geometry_msgs/PoseStamped",
            "/mujoco/marker1/pose": "geometry_msgs/PoseStamped",
            "/mujoco/marker2/pose": "geometry_msgs/PoseStamped"
        })

    log_robot.info(f"检查ROS话题 ({len(topics)}个):")
    log_robot.info("=" * 50)
        
    available = 0
    for topic, msg_type in topics.items():
        try:
            # 动态导入消息类型
            if msg_type == "geometry_msgs/PoseStamped":
                from geometry_msgs.msg import PoseStamped
                msg_class = PoseStamped
            else:
                raise ValueError(f"Unsupported message type: {msg_type}")
            
            # 检查话题
            start_time = time.time()
            rospy.wait_for_message(topic, msg_class, timeout=1.0)
            response_time = time.time() - start_time
            
            log_robot.info(f"✅ {topic} ({response_time:.3f}s)")
            available += 1
            
        except Exception as e:
            log_robot.warning(f"❌ {topic}: {str(e)[:50]}...")
    
    log_robot.info("=" * 50)
    log_robot.info(f"结果: {available}/{len(topics)} 个话题可用")
    return available == len(topics)

def main(config_path: str, episode: int):
    # load config
    cfg = load_inference_config(config_path)
    use_delta = cfg.use_delta
    # eval_episodes = cfg.eval_episodes
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

    # create json file
    json_file_path = output_directory / "evaluation_autotest.json"

    # set seed
    set_seed(seed=seed)

    # Select your device
    device = torch.device(cfg.device)

    policy = setup_policy(pretrained_path, policy_type, device)

    # Initialize evaluation environment to render two observation types:
    # an image of the scene and state/position of the agent.
    max_episode_steps = cfg.max_episode_steps
    env = gym.make(
        env_name,
        max_episode_steps=max_episode_steps,
        config_path=config_path,
    )

    rospy.Subscriber("/simulator/success", Bool, env_success_callback)
    start_service = rospy.ServiceProxy('/simulator/start', Trigger)
    if "task1" in task:
        rospy.Subscriber("/mujoco/box_grab/pose", PoseStamped, box_grab_callback)
        rospy.Subscriber("/mujoco/marker1/pose", PoseStamped, marker1_callback)
        rospy.Subscriber("/mujoco/marker2/pose", PoseStamped, marker2_callback)
    
    check_rostopics(task)

    episode_record = {
        "episode_index": episode,
        "marker1_position": latest_marker1_position,
        "marker2_position": latest_marker2_position,
        "timestamp": datetime.datetime.now().isoformat(),
    }

    # We can verify that the shapes of the features expected by the policy match the ones from the observations
    # produced by the environment
    log_model.info(f"policy.config.input_features: {policy.config.input_features}")
    log_robot.info(f"env.observation_space: {env.observation_space}")

    # Similarly, we can check that the actions produced by the policy will match the actions expected by the
    # environment
    log_model.info(f"policy.config.output_features: {policy.config.output_features}")
    log_robot.info(f"env.action_space: {env.action_space}")

    # Reset the policy and environments to prepare for rollout
    policy.reset()
    numpy_observation, info = env.reset(seed=seed)
    start_service(TriggerRequest())

    # Prepare to collect every rewards and all the frames of the episode,
    # from initial state to final state.
    rewards = []
    # Render frame of the initial state
    # frames.append(env.render())
    cam_keys = [k for k in numpy_observation.keys() if "images" in k or "depth" in k]
    frame_map = {k: [] for k in cam_keys}

    steps_records = []

    step = 0
    done = False
    while not done:
        # log_robot.info(f"random: {np.random.random()}")
        # --- Pause support: block here if pause_flag is set ---
        if not check_control_signals():
            log_robot.info("🛑 收到停止信号，退出机械臂运动")
            sys.exit(1)
        
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

        # Record step data
        js = numpy_observation.get("observation.state", None)
        joint_list = js.tolist() if isinstance(js, np.ndarray) else None
        steps_records.append({
            "object_position": latest_object_position,
            "joint_state": joint_list,
        })

        # 相机帧记录
        for k in cam_keys:
            frame_map[k].append(observation[k].squeeze(0).cpu().numpy().transpose(1, 2, 0))

        # The rollout is considered done when the success state is reached (i.e. terminated is True),
        # or the maximum number of iterations is reached (i.e. truncated is True)
        done = terminated | truncated | done
        done = done or success_evt.is_set()
        step += 1

        end_time = time.time()
        log_model.info(f"Step {step} time: {end_time - start_time:.3f}s")
    
    # Get the speed of environment (i.e. its number of frames per second).
    fps = env.ros_rate
    
    for cam in cam_keys:
        frames = frame_map[cam]
        output_path = output_directory / f"rollout_{episode}_{cam}.mp4"
        imageio.mimsave(str(output_path), frames, fps=fps)

    # Build and append episode record
    success = success_evt.is_set()

    episode_record.update({
        "success": bool(success),
        "step_count": len(steps_records),
        "steps": steps_records,
    })

    # Load existing file, append, and save
    data = {}
    if json_file_path.exists():
        try:
            with open(json_file_path, "r") as f:
                data = json.load(f)
        except Exception:
            data = {}

    if "episodes" not in data:
        data = {"task": task, "episode_num": 0, "episodes": []}

    data["episodes"].append(episode_record)
    data["episode_num"] = len(data["episodes"])

    with open(json_file_path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    sys.exit(0 if success else 1)  # 返回给父进程

def kuavo_eval(config_path: Path, episode: int):
    main(config_path, episode)