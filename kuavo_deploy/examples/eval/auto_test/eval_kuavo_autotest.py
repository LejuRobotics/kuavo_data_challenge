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
from omegaconf import DictConfig, ListConfig, OmegaConf
from torchvision.transforms.functional import to_tensor
from std_msgs.msg import Bool
import rospy
import threading
import traceback
import json

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

init_evt = threading.Event()
def env_init_service(req):
    log_robot.info(f"env_init_callback! req = {req}")
    init_evt.set()
    return TriggerResponse(success=True, message="Env init successful")

def safe_reset_service(reset_service) -> None:
    """安全重置服务"""
    try:
        # 调用重置服务
        response = reset_service(TriggerRequest())
        if response.success:
            log_robot.info(f"Reset service successful: {response.message}")
        else:
            log_robot.warning(f"Reset service failed: {response.message}")
    except rospy.ServiceException as e:
        log_robot.error(f"Reset service exception: {e}")

def kuavo_eval_autotest(config_path: Path):
    """执行自动测试"""
    cfg = load_inference_config(config_path)
    task = cfg.task
    method = cfg.method
    timestamp = cfg.timestamp
    epoch = cfg.epoch
    eval_episodes = cfg.eval_episodes

    output_directory = Path(f"outputs/eval/{task}/{method}/{timestamp}/epoch{epoch}")
    # mkdir
    output_directory.mkdir(parents=True, exist_ok=True)

    # Log evaluation results
    log_file_path = output_directory / "evaluation_autotest.log"
    with log_file_path.open("w") as log_file:
        log_file.write(f"Evaluation Timestamp: {datetime.datetime.now()}\n")
        log_file.write(f"Total Episodes: {eval_episodes}\n")

    # create json file
    json_file_path = output_directory / "evaluation_autotest.json"
    episode_results = []
    episode_data = {
        "task":task,
        "episode_num": 0,
        "episodes": [],
    }
    episode_results.append(episode_data)
    with json_file_path.open("w", encoding="utf-8") as json_file:
        json.dump(episode_results, json_file, indent=2, ensure_ascii=False)

    # Ros service
    init_service = rospy.Service("/simulator/init", Trigger, env_init_service)
    # wait for first init
    while not init_evt.is_set():
        log_robot.info("Waiting for first env init...")
        if not check_control_signals():
            log_robot.info("🛑 收到停止信号，退出机械臂运动")
            return
        time.sleep(5)

    # Ros service
    reset_service = rospy.ServiceProxy('/simulator/reset', Trigger)
    safe_reset_service(reset_service)

    success_count = 0
    for episode in range(eval_episodes):
        init_evt.clear()

        while not init_evt.is_set():
            log_robot.info("Waiting for env init...")
            if not check_control_signals():
                log_robot.info("🛑 收到停止信号，退出机械臂运动")
                return
            time.sleep(5)

        result = subprocess.run(['python', '-c', f'from kuavo_deploy.examples.eval.auto_test.eval_kuavo import kuavo_eval; kuavo_eval("{config_path}", {episode})'], 
                                capture_output=False, text=True)
        log_robot.info(f"Episode {episode+1} completed with return code: {result.returncode}")

        # 新增：记录episode结果到JSON
        episode_end_time = datetime.datetime.now().isoformat()
        is_success = result.returncode == 0
        if is_success:
            success_count += 1
            log_model.info(f"✅ Episode {episode+1}: Success!")
        else:
            log_model.info(f"❌ Episode {episode+1}: Failed!")

        safe_reset_service(reset_service)

        with log_file_path.open("a") as log_file:
            log_file.write("\n")
            log_file.write(f"Success Count: {success_count} / Already eval episodes: {episode+1}")

    # Display final statistics
    log_model.info("\n" + "="*50)
    log_model.info(f"🎯 Evaluation completed!")
    log_model.info(f"📊 Success count: {success_count}/{eval_episodes}")
    log_model.info(f"📈 Success rate: {success_count / eval_episodes:.2%}")
    log_model.info(f"📁 Videos and logs saved to: {output_directory}")
    log_model.info(f"📁 JSON results saved to: {json_file_path}")
    log_model.info("="*50)

if __name__ == "__main__":
    config_path = Path("test.yaml")
    env = gym.make(
        "Kuavo-Real",
        max_episode_steps=150,
        config_path=config_path,
    )
