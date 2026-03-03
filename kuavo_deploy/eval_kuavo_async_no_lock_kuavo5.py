#!/usr/bin/env python

"""
实机异步推理脚本：使用真实 ROS 环境获取观测，进行异步 ACT 推理。

This script demonstrates:
1. Using real ROS environment (KuavoRealEnv) to get observations
2. Asynchronous ACT inference without RTC
3. Real-time action execution on physical robot

Usage:
    python kuavo_deploy/eval_kuavo_async_online.py \
        --device=cuda \
        --fps=10.0 \
        --task="pick the fruit" \
        --duration=300
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Ensure patched FeatureType/PolicyFeature (with DEPTH support) are registered
from lerobot_patches import custom_patches  # noqa: F401  (side-effect import)

import logging
import time
import traceback
from dataclasses import dataclass, field
from threading import Event, Thread
from queue import Queue, Empty
from typing import Optional

import torch
import numpy as np

from lerobot.configs import parser
from lerobot.policies.factory import make_pre_post_processors

from lerobot.utils.hub import HubMixin
from lerobot.utils.utils import init_logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from pathlib import Path
from kuavo_train.wrapper.policy.act.ACTPolicyWrapper import CustomACTPolicyWrapper
from kuavo_deploy.config_kuavo5 import load_kuavo_config
from kuavo_deploy.kuavo_env.KuavoRealEnv import KuavoRealEnv
from kuavo_deploy.src.scripts.script import ArmMove
from PIL import Image
import cv2


class ActionBuffer:
    """
    简单的动作缓冲区，用于存储和获取动作。
    不使用RTC，只是简单的FIFO队列。
    """
    def __init__(self, maxsize: int = 18):
        self.queue = Queue(maxsize=maxsize)
        self.maxsize = maxsize
        self.actions_executed = 0  # 记录已执行的动作数量
    
    def put(self, actions: torch.Tensor, replace: bool = False):
        """添加动作块到缓冲区
        
        Args:
            actions: (chunk_size, action_dim) 动作块
            replace: 如果为True，清空队列后用新动作块替换；如果为False，追加到队列末尾
        """
        if replace:
            # 清空队列，用新动作块替换
            self.clear()
        
        # actions: (chunk_size, action_dim)
        for action in actions:
            try:
                self.queue.put(action, block=False)
            except:
                # 队列满时，丢弃最旧的动作
                try:
                    self.queue.get_nowait()
                    self.queue.put(action, block=False)
                except:
                    pass
    
    def get(self, timeout: float = 0.1) -> Optional[torch.Tensor]:
        """从缓冲区获取一个动作"""
        try:
            action = self.queue.get(timeout=timeout)
            self.actions_executed += 1
            return action
        except Empty:
            return None
    
    def qsize(self) -> int:
        """返回队列大小"""
        return self.queue.qsize()
    
    def get_actions_executed(self) -> int:
        """返回已执行的动作数量"""
        return self.actions_executed
    
    def reset_execution_count(self):
        """重置执行计数（用于同步观测获取）"""
        self.actions_executed = 0
    
    def clear(self):
        """清空缓冲区"""
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except Empty:
                break


@dataclass
class ACTAsyncRealDemoConfig(HubMixin):
    """Configuration for ACT async demo on real robot."""

    # Demo parameters
    duration: float = 30.0  # Duration to run the demo (seconds)
    fps: float = 10.0  # Action execution frequency (Hz)

    # Compute device
    device: str = field(default="cuda", metadata={"help": "Device to run on (cuda, cpu, auto)"})

    # Action buffer configuration
    action_buffer_size: int = 18  # Maximum number of actions in buffer
    min_buffer_size: int = 4  # Minimum buffer size before requesting new actions
    obs_update_interval: int = 1  # Update observation every N executed actions (should match action execution frequency)

    # Task to execute
    task: str = field(default="Depalletize the box", metadata={"help": "Task to execute"})

    # Model optimization
    use_torch_compile: bool = field(
        default=False,
        metadata={"help": "Use torch.compile for faster inference (PyTorch 2.0+)"},
    )

    torch_compile_backend: str = field(
        default="inductor",
        metadata={"help": "Backend for torch.compile (inductor, aot_eager, cudagraphs)"},
    )

    torch_compile_mode: str = field(
        default="default",
        metadata={"help": "Compilation mode (default, reduce-overhead, max-autotune)"},
    )


def get_actions_async(
    policy,
    preprocessor,
    postprocessor,
    env: KuavoRealEnv,
    action_buffer: ActionBuffer,
    shutdown_event: Event,
    fps: float,
    min_buffer_size: int,
    device: torch.device,
    task: str,
    obs_update_interval: int = 1,
):
    """Thread function to request action chunks from the policy asynchronously.

    Args:
        policy: The ACT policy instance
        env: The real robot environment instance
        action_buffer: Buffer to put new action chunks
        shutdown_event: Event to signal shutdown
        fps: Action execution frequency
        min_buffer_size: Minimum buffer size before requesting new actions
        device: Device to run inference on
        task: Task description
        # visualize_obs: Whether to save observation visualizations
        # visualize_obs_interval: Save visualization every N inference calls
        # show_realtime_obs: Whether to show real-time visualization windows
        obs_update_interval: Update observation every N executed actions
    """
    try:
        logger.info("[GET_ACTIONS] Starting async inference thread")
        print("[GET_ACTIONS] Starting async inference thread", flush=True)
        
        inference_step = 0
        # vis_output_dir = Path("debug_obs") if visualize_obs else None

        while not shutdown_event.is_set():
            # 异步推理的正确逻辑（online版本）：
            # 1. 只在buffer快空时，才获取新观测并推理新动作块
            # 2. 不需要在不推理时获取观测，因为观测会实时更新（从ROS环境获取）
            # 3. 推理是异步的：在执行当前动作块的同时，可以基于新观测推理下一个动作块
            # 4. 这样可以在执行旧动作块时，后台推理新动作块，减少停顿
            
            actions_executed = action_buffer.get_actions_executed()
            buffer_size = action_buffer.qsize()
            
            # 需要生成新动作的条件：只在buffer快空时触发推理
            # online版本不需要在不推理时获取观测，因为观测会实时更新
            need_new_actions = buffer_size <= min_buffer_size
            
            if need_new_actions:
                trigger_reason = "buffer_low"
                logger.info(
                    f"[GET_ACTIONS] Triggering inference: reason={trigger_reason}, "
                    f"buffer_size={buffer_size}, actions_executed={actions_executed}, generating new actions..."
                )
                print(
                    f"[GET_ACTIONS] Triggering inference: reason={trigger_reason}, "
                    f"buffer_size={buffer_size}, actions_executed={actions_executed}",
                    flush=True
                )
                
                # 开始计时
                inference_start_time = time.time()
                
                # Get observation from real robot environment
                # online版本：只在需要推理时获取观测，因为观测会实时更新
                obs_fetch_start = time.time()
                obs = env.get_obs()  # Returns dict with observation.images.*, observation.depth*, observation.state
                obs_fetch_time = time.time() - obs_fetch_start
                
                # Preprocess observation
                preprocess_start = time.time()
                processed_observation = preprocessor(obs)
                preprocess_time = time.time() - preprocess_start
                
                # Generate action chunk (ACT without RTC)
                inference_model_start = time.time()
                with torch.no_grad():
                    actions = policy.predict_action_chunk(processed_observation)
                inference_model_time = time.time() - inference_model_start
                print(f"sleep time: {max(0, 0.03 - inference_model_time)}")
                time.sleep(max(0, 0.03 - inference_model_time))

                # actions shape: (B, chunk_size, action_dim) or (chunk_size, action_dim)
                if actions.dim() == 3:
                    actions = actions.squeeze(0)  # Remove batch dimension if present
                
                # Postprocess actions
                postprocess_start = time.time()
                chunk_size = actions.shape[0]
                processed_actions = []
                for i in range(chunk_size):
                    single_action = actions[i:i+1]  # (1, action_dim)
                    processed_action = postprocessor(single_action)
                    processed_actions.append(processed_action.squeeze(0))  # (action_dim,)
                
                # Stack to (chunk_size, action_dim)
                postprocessed_actions = torch.stack(processed_actions, dim=0)
                postprocess_time = time.time() - postprocess_start
                
                # Add to buffer: 基于新观测的动作块应该替换旧队列
                # 这样可以确保执行的动作总是基于最新的观测
                action_buffer.put(postprocessed_actions[3:21], replace=True)
                inference_step += 1
                
                # 计算总时间
                total_inference_time = time.time() - inference_start_time
                
                # 打印时间开销
                timing_msg = (
                    f"[GET_ACTIONS] Inference timing (step {inference_step}): "
                    f"obs_fetch={obs_fetch_time*1000:.2f}ms, "
                    f"preprocess={preprocess_time*1000:.2f}ms, "
                    f"model_inference={inference_model_time*1000:.2f}ms, "
                    f"postprocess={postprocess_time*1000:.2f}ms, "
                    f"total={total_inference_time*1000:.2f}ms, "
                    f"chunk_size={chunk_size}, buffer_size={action_buffer.qsize()}"
                )
                logger.info(timing_msg)
                print(timing_msg, flush=True)
            else:
                # Small sleep to prevent busy waiting
                time.sleep(0.01)

        logger.info("[GET_ACTIONS] Async inference thread shutting down")
    except Exception as e:
        print("\n" + "="*80, file=sys.stderr, flush=True)
        print("[GET_ACTIONS] FATAL EXCEPTION", file=sys.stderr, flush=True)
        traceback.print_exc()
        print("="*80 + "\n", file=sys.stderr, flush=True)
        shutdown_event.set()
        return


def actor_control(
    env: KuavoRealEnv,
    action_buffer: ActionBuffer,
    shutdown_event: Event,
    cfg: ACTAsyncRealDemoConfig,
):
    """Thread function to execute actions on the robot.

    Args:
        env: The robot environment instance
        action_buffer: Buffer to get actions from
        shutdown_event: Event to signal shutdown
        cfg: Demo configuration
    """
    try:
        logger.info("[ACTOR] Starting actor thread")

        action_count = 0

        while not shutdown_event.is_set():
            # Get an action from the buffer
            action = action_buffer.get(timeout=0.1)

            if action is not None:
                action = action.cpu()
                # KuavoRealEnv uses step(action) to execute actions
                env.step(action.numpy())
                action_count += 1
                logger.info(f"[ACTOR] Executed action #{action_count}")
            else:
                logger.warn("[ACTOR] No action available in buffer")

        logger.info(f"[ACTOR] Actor thread shutting down. Total actions executed: {action_count}")
    except Exception as e:
        print("\n" + "="*80, file=sys.stderr, flush=True)
        print("[ACTOR] FATAL EXCEPTION", file=sys.stderr, flush=True)
        traceback.print_exc()
        print("="*80 + "\n", file=sys.stderr, flush=True)
        shutdown_event.set()
        return


def _apply_torch_compile(policy, cfg: ACTAsyncRealDemoConfig):
    """
    Apply torch.compile to the policy's predict_action_chunk method for faster inference.
    
    Args:
        policy: The ACT policy instance
        cfg: Demo configuration
    
    Returns:
        Policy with compiled predict_action_chunk method
    """
    try:
        # Check if torch.compile is available (PyTorch 2.0+)
        if not hasattr(torch, "compile"):
            logger.warning(
                f"torch.compile is not available. Requires PyTorch 2.0+. "
                f"Current version: {torch.__version__}. Skipping compilation."
            )
            return policy

        logger.info("Applying torch.compile to predict_action_chunk...")
        logger.info(f"  Backend: {cfg.torch_compile_backend}")
        logger.info(f"  Mode: {cfg.torch_compile_mode}")

        compile_kwargs = {
            "backend": cfg.torch_compile_backend,
            "mode": cfg.torch_compile_mode,
        }

        original_method = policy.predict_action_chunk
        compiled_method = torch.compile(original_method, **compile_kwargs)
        policy.predict_action_chunk = compiled_method
        logger.info("✓ Successfully compiled predict_action_chunk")

    except Exception as e:
        logger.error(f"Failed to apply torch.compile: {e}")
        logger.warning("Continuing without torch.compile")

    return policy


@parser.wrap()
def demo_cli(cfg: ACTAsyncRealDemoConfig):
    """Main entry point for ACT async demo on real robot."""

    # 初始化日志
    init_logging()
    logger.info(f"[MAIN] Using device: {cfg.device}")

    # 设置信号处理器
    from lerobot.rl.process import ProcessSignalHandler
    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    shutdown_event = signal_handler.shutdown_event

    # 加载 Kuavo 配置（YAML），与 eval_kuavo.py / real_single_test.py 保持一致
    kuavo_config = load_kuavo_config()

    # 创建 Kuavo 实机环境
    env = KuavoRealEnv(kuavo_config)
    # arm_move = ArmMove(kuavo_config)
    logger.info("[MAIN] KuavoRealEnv initialized")

    # ---------------- 加载 policy（参考 real_single_test.py） ----------------
    inf_cfg = kuavo_config.inference
    device = torch.device(inf_cfg.device)

    # 与 real_single_test 相同的路径规则：outputs/train/{task}/{method}/{timestamp}/epoch{epoch}
    pretrained_path = Path(f"outputs/train/{inf_cfg.task}/{inf_cfg.method}/{inf_cfg.timestamp}/epoch{inf_cfg.epoch}")
    logger.info(f"[MAIN] Loading ACT policy from {pretrained_path}")
    policy = CustomACTPolicyWrapper.from_pretrained(pretrained_path, strict=True)

    policy.temporal_ensemble_coeff = None
    policy.temporal_ensemble = None
    policy.n_action_steps = 18
    
    policy.eval()
    policy.to(device)
    policy.reset()
    logger.info(f"[MAIN] Policy loaded and set to eval mode on {device}")

    # 加载预处理 / 后处理（参考 real_single_test.py）
    base_path = Path(str(pretrained_path).split("/epoch", 1)[0])
    preprocessor, postprocessor = make_pre_post_processors(None, base_path)

    if cfg.use_torch_compile:
        policy = _apply_torch_compile(policy, cfg)

    # 初始化 ActionBuffer (不使用RTC的ActionQueue)
    action_buffer = ActionBuffer(maxsize=cfg.action_buffer_size)
    logger.info("[MAIN] ActionBuffer created")

    # 外层循环：可以多次运行推理
    while True:
        try:
            # 通过环境自身 reset 完成机器人初始化（进入外部控制模式、复位头和夹爪等）
            env.reset()
            # arm_move.go()
            # 交互控制：等待用户按回车开始
            print("\n" + "="*80)
            print("[MAIN] Robot initialized. Press Enter to start inference (or Ctrl+C to exit)...")
            print("="*80)
            input()
            
            # 初始化
            shutdown_event.clear()
            policy.reset()
            action_buffer.clear()  # 清空动作缓冲区

            # 启动线程
            get_actions_thread = Thread(
                target=get_actions_async,
                args=(policy, preprocessor, postprocessor, env, action_buffer, shutdown_event, cfg.fps, cfg.min_buffer_size, device, cfg.task, cfg.obs_update_interval),
                daemon=True,
                name="GetActions"
            )
            actor_thread = Thread(
                target=actor_control, 
                args=(env, action_buffer, shutdown_event, cfg), 
                daemon=True, 
                name="Actor"
            )
            get_actions_thread.start()
            actor_thread.start()

            logger.info("[MAIN] Threads started. Running demo...")
            print("[MAIN] Demo running. Press 'q' and Enter to stop (or Ctrl+C to exit)...")
            
            # 交互控制：等待用户输入 'q' 结束当前推理
            while True:
                try:
                    user_input = input().strip().lower()
                    if user_input == 'q':
                        logger.info("[MAIN] User requested stop (q)")
                        shutdown_event.set()
                        break
                except (EOFError, KeyboardInterrupt):
                    logger.info("[MAIN] Interrupted, shutting down...")
                    shutdown_event.set()
                    raise  # 重新抛出异常，退出外层循环

            get_actions_thread.join(timeout=5)
            actor_thread.join(timeout=5)
            logger.info("[MAIN] Demo finished. Ready for next run.")
            
        except KeyboardInterrupt:
            logger.info("[MAIN] Keyboard interrupt, exiting...")
            shutdown_event.set()
            break
        except Exception as e:
            logger.error(f"[MAIN] Error during demo: {e}")
            traceback.print_exc()
            shutdown_event.set()
            break
    
    # Clean up
    env.close()


if __name__ == "__main__":
    demo_cli()
    logging.info("ACT async demo (real robot) finished")
