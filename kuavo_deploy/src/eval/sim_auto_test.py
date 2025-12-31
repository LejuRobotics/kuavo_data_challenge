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
import sys,os
import gc
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
from kuavo_train.wrapper.policy.act.ACTPolicyWrapper import CustomACTPolicyWrapper
from lerobot.policies.act.modeling_act import ACTPolicy
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
from kuavo_deploy.config import KuavoConfig
from kuavo_deploy.utils.logging_utils import setup_logger
from kuavo_deploy.kuavo_service.client import PolicyClient
from lerobot.policies.factory import make_pre_post_processors
log_model = setup_logger("model")
log_robot = setup_logger("robot")

from kuavo_deploy.kuavo_env.KuavoSimEnv import KuavoSimEnv
from kuavo_deploy.kuavo_env.KuavoRealEnv import KuavoRealEnv
from kuavo_deploy.utils.ros_manager import ROSManager


init_evt = threading.Event()
pause_flag = threading.Event()
stop_flag = threading.Event()
success_evt = threading.Event()

def env_init_service(req):
    log_robot.info(f"env_init_callback! req = {req}")
    init_evt.set()
    return TriggerResponse(success=True, message="Env init successful")

def pause_callback(msg):
    if msg.data:
        pause_flag.set()
    else:
        pause_flag.clear()

def stop_callback(msg):
    if msg.data:
        stop_flag.set()

def env_success_callback(msg):
    # log_model.info("env_success_callback!")
    if msg.data:
        success_evt.set()


pause_sub = rospy.Subscriber('/kuavo/pause_state', Bool, pause_callback, queue_size=10)
stop_sub = rospy.Subscriber('/kuavo/stop_state', Bool, stop_callback, queue_size=10)

def safe_reset_service(reset_service) -> None:
    """安全重置服务 Safe resetting service"""
    try:
        # 调用重置服务 Reset
        response = reset_service(TriggerRequest())
        if response.success:
            log_robot.info(f"Reset service successful: {response.message}")
        else:
            log_robot.warning(f"Reset service failed: {response.message}")
    except rospy.ServiceException as e:
        log_robot.error(f"Reset service exception: {e}")

def check_control_signals():
    """检查控制信号 Check control signal"""
    # 检查暂停状态 Pause state
    while pause_flag.is_set():
        log_robot.info("🔄 Robot arm motion paused")
        time.sleep(0.1)
        if stop_flag.is_set():
            log_robot.info("🛑 Robot arm motion stopped")
            return False
    
    # 检查是否需要停止 Whether it needs to be stopped
    if stop_flag.is_set():
        log_robot.info("🛑 Stop signal detected, exiting robot arm motion")
        return False
        
    return True  # 正常继续 Continue


    
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
        policy = CustomACTPolicyWrapper.from_pretrained(Path(pretrained_path),strict=True)
    elif policy_type == 'client':
        policy = PolicyClient()
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

def run_single_episode(config, policy, preprocessor, postprocessor, episode, output_directory):
    """运行单个episode Running a single episode"""
    cfg = config.inference
    seed = cfg.seed
    task = cfg.task
    # Initialize environment
    env = gym.make(
        config.env.env_name,
        max_episode_steps=cfg.max_episode_steps,
        config=config,
    )

    run_single_ros_manager = ROSManager()
    # Setup ROS subscribers and services
    run_single_ros_manager.register_subscriber("/simulator/success", Bool, env_success_callback)

    # max_episode_steps = cfg.max_episode_steps

    start_service = rospy.ServiceProxy('/simulator/start', Trigger)


    if cfg.policy_type != 'client':
        log_model.info(f"policy.config.input_features: {policy.config.input_features}")
        log_robot.info(f"env.observation_space: {env.observation_space}")
        log_model.info(f"policy.config.output_features: {policy.config.output_features}")
        log_robot.info(f"env.action_space: {env.action_space}")

    # Reset the policy and environments to prepare for rollout
    policy.reset()
    observation, info = env.reset(seed=seed)
    # first_img =  (observation["observation.images.head_cam_h"].squeeze().permute(1,2,0).numpy()*255).astype(np.uint8)
    
    # import cv2
    # first_img = cv2.cvtColor(first_img,cv2.COLOR_RGB2BGR)
    # cv2.imwrite( "obs.png", first_img)
    # raise ValueError("stop for debug!")
    start_service(TriggerRequest())

    # Prepare to collect every rewards and all the frames of the episode,
    # from initial state to final state.
    rewards = []
    cam_keys = [k for k in observation.keys() if "images" in k or "depth" in k]

    frame_temp_dirs = {}
    for k in cam_keys:
        temp_dir = output_directory / f"temp_frames_{episode}_{k}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        frame_temp_dirs[k] = temp_dir


    average_exec_time = 0
    average_action_infer_time = 0
    average_step_time = 0

    step = 0
    done = False
    while not done:
        # --- Pause support: block here if pause_flag is set ---
        if not check_control_signals():
            log_robot.info("🛑 Stop signal detected, exiting robot arm motion")
            return 0
        
        start_time = time.time()
        observation = preprocessor(observation)
        with torch.inference_mode():
            action = policy.select_action(observation)
        log_model.info(f"Step {step}: predict action {action}")
        action = postprocessor(action)
        # print(f"action: {action}, action.shape: {action.shape}, action min: {action.min()}, action max: {action.max()}")
        action_infer_time = time.time()
        log_model.info(f"episode {episode}, step {step}, action infer time: {action_infer_time - start_time:.3f}s")
        average_action_infer_time += action_infer_time - start_time

        numpy_action = action.squeeze(0).cpu().numpy()

        log_model.info(f"Step {step}: Executing action {numpy_action}")
        observation, reward, terminated, truncated, info = env.step(numpy_action)

        exec_time = time.time()
        log_model.debug(f"step {step}: exec time: {exec_time - action_infer_time:.3f}s")
        average_exec_time += exec_time - action_infer_time
        
        rewards.append(reward)

        for k in cam_keys:
            frame_path = frame_temp_dirs[k] / f"frame_{step:04d}.png"
            img = (observation[k].squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            if img.shape[-1] == 1:
                img = img.squeeze(-1)
            imageio.imwrite(str(frame_path), img)

        # The rollout is considered done when the success state is reached (i.e. terminated is True),
        # or the maximum number of iterations is reached (i.e. truncated is True)
        done = terminated | truncated | done
        done = done or success_evt.is_set()
        step += 1

        end_time = time.time()
        log_model.debug(f"Step {step} time: {end_time - start_time:.3f}s")
        average_step_time += end_time - start_time
    
    # Get the speed of environment (i.e. its number of frames per second).
    fps = env.unwrapped.ros_rate

    log_model.info(f"average exec time: {average_exec_time / step:.3f}s")
    log_model.info(f"average action infer time: {average_action_infer_time / step:.3f}s")
    log_model.info(f"average step time: {average_step_time / step:.3f}s")
    log_model.info(f"average sleep time: {env.unwrapped.average_sleep_time / step:.3f}s")
    
    for cam in cam_keys:
        temp_dir = frame_temp_dirs[cam]
        frame_files = sorted(temp_dir.glob("frame_*.png"))
        frames = [imageio.imread(str(f)) for f in frame_files]
        output_path = output_directory / f"rollout_{episode}_{cam}.mp4"
        imageio.mimsave(str(output_path), frames, fps=fps)
        

        for f in frame_files:
            f.unlink()
        temp_dir.rmdir()
        
        del frames

    success = success_evt.is_set()
    
    env.close()
    run_single_ros_manager.close()
    
    del rewards
    del observation
    del env
    del run_single_ros_manager
    
    gc.collect()
    torch.cuda.empty_cache()
    
    return 1 if success else 0


def kuavo_eval_autotest(config: KuavoConfig):
    """执行自动测试 Auto testing script"""
    cfg = config.inference
    task = cfg.task
    method = cfg.method
    timestamp = cfg.timestamp
    epoch = cfg.epoch
    eval_episodes = cfg.eval_episodes
    seed = cfg.seed
    policy_type = cfg.policy_type

    # Setup paths
    pretrained_path = Path(f"outputs/train/{task}/{method}/{timestamp}/epoch{epoch}")
    output_directory = Path(f"outputs/eval/{task}/{method}/{timestamp}/epoch{epoch}")
    output_directory.mkdir(parents=True, exist_ok=True)

    # Log evaluation results
    log_file_path = output_directory / "evaluation_autotest.log"
    
    with log_file_path.open("w") as log_file:
        log_file.write(f"Evaluation Timestamp: {datetime.datetime.now()}\n")
        log_file.write(f"Total Episodes: {eval_episodes}\n")
    
    
    # Setup policy and environment (只加载一次)
    set_seed(seed)
    device = torch.device(cfg.device)
    policy = setup_policy(pretrained_path, policy_type, device)
    preprocessor, postprocessor = make_pre_post_processors(None, Path(str(pretrained_path).split("/epoch", 1)[0]))
    
    # first reset
    reset_service = rospy.ServiceProxy('/simulator/reset', Trigger)
    # Ros service
    init_service = rospy.Service("/simulator/init", Trigger, env_init_service)


    wait_times = 8
    while not init_evt.is_set():
        log_robot.info("Waiting for first env init...")
        if not check_control_signals():
            log_robot.info("🛑 Stop signal detected, exiting robot arm motion")
            return
        time.sleep(1)
        wait_times -= 1
        if wait_times <=0:
            break
    safe_reset_service(reset_service)
    init_evt.clear()

    success_count = 0
    for episode in range(eval_episodes):

        while not init_evt.is_set():
            log_robot.info("Waiting for env init...")
            if not check_control_signals():
                log_robot.info("🛑 Stop signal detected, exiting robot arm motion")
                return
            time.sleep(1)
        try:
            result = run_single_episode(config, policy, preprocessor, postprocessor, episode, output_directory)
            log_robot.info(f"Episode {episode+1} completed with return code: {result}")
            
            # 重置policy状态，清理缓存 Reset policy, clear cache
            policy.reset()
            
            # 强制垃圾回收和GPU缓存清理 Force garbage collection and GPU cache cleaning
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            log_robot.error(f"Exception during episode {episode+1}: {e}")
            log_robot.error(traceback.format_exc())
            result = 0  # Treat as failure
            safe_reset_service(reset_service)
            init_evt.clear()
            success_evt.clear()
            
            # 异常情况下也要清理内存 Garbage collection even under an exception
            gc.collect()
            torch.cuda.empty_cache()
            break

        # 记录episode结果 Record episode result
        episode_end_time = datetime.datetime.now().isoformat()
        is_success = result == 1
        if is_success:
            success_count += 1
            log_model.info(f"✅ Episode {episode+1}: Success!")
        else:
            log_model.info(f"❌ Episode {episode+1}: Failed!")



        with log_file_path.open("a") as log_file:
            log_file.write("\n")
            log_file.write(f"Success Count: {success_count} / Already eval episodes: {episode+1}")
    
        safe_reset_service(reset_service)
        init_evt.clear()
        success_evt.clear()
    

    # Display final statistics
    log_model.info("\n" + "="*50)
    log_model.info(f"🎯 Evaluation completed!")
    log_model.info(f"📊 Success count: {success_count}/{eval_episodes}")
    log_model.info(f"📈 Success rate: {success_count / eval_episodes:.2%}")
    log_model.info(f"📁 Videos and logs saved to: {output_directory}")
    log_model.info("="*50)
    init_service.shutdown()
    pause_sub.unregister()
    stop_sub.unregister()

