"""
机器人控制示例程序
提供机械臂运动控制、轨迹回放等功能

使用示例:
  python scripts_auto_test.py --task auto_test --config /path/to/custom_config.yaml
"""

import rospy
import rosbag
import time
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

from std_srvs.srv import Trigger, TriggerRequest, TriggerResponse

from kuavo_deploy.utils.logging_utils import setup_logger
from kuavo_deploy.kuavo_env.KuavoBaseRosEnv import KuavoBaseRosEnv
from kuavo_deploy.config import load_kuavo_config, KuavoConfig
import gymnasium as gym

import numpy as np
import signal
import sys,os
import threading
import subprocess
import traceback

from std_msgs.msg import Bool

# 配置日志
log_model = setup_logger("model", "DEBUG")  # 网络日志
log_robot = setup_logger("robot", "DEBUG")  # 机器人日志

# 控制变量
class ArmMoveController:
    def __init__(self):
        self.paused = False
        self.should_stop = False
        self.lock = threading.Lock()
        
    def pause(self):
        with self.lock:
            self.paused = True
            log_robot.info("🔄 机械臂运动已暂停")
    
    def resume(self):
        with self.lock:
            self.paused = False
            log_robot.info("▶️ 机械臂运动已恢复")
    
    def stop(self):
        with self.lock:
            self.should_stop = True
            log_robot.info("⏹️ 机械臂运动已停止")
    
    def is_paused(self):
        with self.lock:
            return self.paused
    
    def should_exit(self):
        with self.lock:
            return self.should_stop

# 控制器实例
arm_controller = ArmMoveController()

# Ros发布暂停/停止信号
pause_pub = rospy.Publisher('/kuavo/pause_state', Bool, queue_size=1)
stop_pub = rospy.Publisher('/kuavo/stop_state', Bool, queue_size=1)

def signal_handler(signum, frame):
    """信号处理器"""
    log_robot.info(f"🔔 收到信号: {signum}")
    if signum == signal.SIGUSR1:  # 暂停/恢复
        if arm_controller.is_paused():
            log_robot.info("🔔 当前状态：已暂停，执行恢复")
            arm_controller.resume()
            pause_pub.publish(False)
        else:
            log_robot.info("🔔 当前状态：运行中，执行暂停")
            arm_controller.pause()
            pause_pub.publish(True)
    elif signum == signal.SIGUSR2:  # 停止
        log_robot.info("�� 执行停止")
        arm_controller.stop()
        stop_pub.publish(True)
    log_robot.info(f"🔔 信号处理完成，当前状态 - 暂停: {arm_controller.is_paused()}, 停止: {arm_controller.should_exit()}")

def setup_signal_handlers():
    """设置信号处理器"""
    signal.signal(signal.SIGUSR1, signal_handler)  # 暂停/恢复
    signal.signal(signal.SIGUSR2, signal_handler)  # 停止
    log_robot.info("📡 信号处理器已设置:")
    log_robot.info("  SIGUSR1 (kill -USR1): 暂停/恢复机械臂运动")
    log_robot.info("  SIGUSR2 (kill -USR2): 停止机械臂运动")

class ArmMove:
    """机械臂运动控制类"""
    
    def __init__(self, config: KuavoConfig):
        """
        初始化机械臂控制
        
        Args:
            bag_path: 轨迹文件路径
        """
        self.config = config

        # 设置信号处理器
        self.shutdown_requested = False
        # 设置信号处理器
        setup_signal_handlers()
        
        # 输出当前进程ID，方便外部控制
        pid = os.getpid()
        log_robot.info(f"🆔 当前进程ID: {pid}")
        log_robot.info(f"💡 使用以下命令控制机械臂运动:")
        log_robot.info(f"   暂停/恢复: kill -USR1 {pid}")
        log_robot.info(f"   停止运动: kill -USR2 {pid}")

        self.inference_config = config.inference

        rospy.init_node('kuavo_deploy', anonymous=True)

    def _check_control_signals(self):
        """检查控制信号"""
        # 检查暂停状态
        while arm_controller.is_paused():
            log_robot.info("🔄 机械臂运动已暂停")
            time.sleep(0.1)
            if arm_controller.should_exit():
                log_robot.info("🛑 机械臂运动被停止")
                return False
        
        # 检查是否需要停止
        if arm_controller.should_exit():
            log_robot.info("🛑 收到停止信号，退出机械臂运动")
            return False
            
        return True  # 正常继续
    

    def auto_test(self) -> None:
        """执行自动测试"""
        from kuavo_deploy.src.eval.sim_auto_test import kuavo_eval_autotest
        kuavo_eval_autotest(config=self.config)
    
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Kuavo机器人控制示例程序",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python scripts_auto_test.py --task auto_test --config /path/to/custom_config.yaml"           # 仿真中自动测试模型，执行eval_episodes次


任务说明:
  auto_test   - 仿真中自动测试模型，执行eval_episodes次
        """
    )
    
    # 必需参数
    parser.add_argument(
        "--task", 
        type=str, 
        required=True,
        choices=["auto_test"],
        help="要执行的任务类型"
    )
    
    # 可选参数
    parser.add_argument(
        "--config", 
        type=str,
        required=True,
        help="配置文件路径(必须指定)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="启用详细输出"
    )
    
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="干运行模式，只显示将要执行的操作但不实际执行"
    )
    
    return parser.parse_args()

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置日志级别
    if args.verbose:
        log_model.setLevel("DEBUG")
        log_robot.setLevel("DEBUG")
    
    # 确定配置文件路径
    config_path = Path(args.config)
    
    log_robot.info(f"使用配置文件: {config_path}")
    log_robot.info(f"执行任务: {args.task}")
    
    config = load_kuavo_config(config_path)
    # 初始化机械臂
    try:
        arm = ArmMove(config)
        log_robot.info("机械臂初始化成功")
    except Exception as e:
        log_robot.error(f"机械臂初始化失败: {e}")
        return
    
    # 干运行模式
    if args.dry_run:
        log_robot.info("=== 干运行模式 ===")
        log_robot.info(f"将要执行的任务: {args.task}")
        log_robot.info("干运行模式结束，未实际执行任何操作")
        return
    
    # 任务映射
    task_map = {
        "auto_test": arm.auto_test,      # 仿真中自动测试模型，执行eval_episodes次
    }
    
    # 执行任务
    try:
        log_robot.info(f"开始执行任务: {args.task}")
        task_map[args.task]()
        log_robot.info(f"任务 {args.task} 执行完成")
    except KeyboardInterrupt:
        log_robot.info("用户中断操作")
    except Exception as e:
        traceback.print_exc()
        log_robot.error(f"执行任务 {args.task} 时发生错误: {e}")

if __name__ == "__main__":
    main()
