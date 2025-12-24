#!/usr/bin/env python3
import rospy
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, JointState
import cv2
import gymnasium as gym
import time
import sys
from kuavo_humanoid_sdk import KuavoSDK, KuavoRobot, KuavoRobotState, DexterousHand
from kuavo_humanoid_sdk.msg.kuavo_msgs.msg import lejuClawCommand
from kuavo_deploy.utils.logging_utils import setup_logger
from kuavo_deploy.config import KuavoConfig
from kuavo_deploy.utils.ros_manager import ROSManager
import traceback
import torch
from torchvision.transforms.functional import to_tensor
from kuavo_deploy.utils.obs_buffer import ObsBuffer
from kuavo_deploy.utils.signal_controller import ControlSignalManager


log_robot = setup_logger("robot")


class KuavoBaseRosEnv(gym.Env):
    """Kuavo机器人ROS环境基类"""

    def __init__(self, config: KuavoConfig):
        self._set_config(config.env)
        
        # 初始化ROS管理器
        self.ros_manager = ROSManager()
        self.control_signal_manager = ControlSignalManager()
        
        # 初始化其他组件
        self.bridge = CvBridge()
        self._set_observation_space()
        self._set_action_space()
        self._init_kuavo_sdk()
        self._set_ros_topics()
        
        # 等待ROS话题初始化
        log_robot.info(f"Inializing done!")
        print(f"Inializing done!")

    def _set_config(self, config_kuavo_env):
        """设置配置参数"""
        self.ros_rate = config_kuavo_env.ros_rate
        self.control_mode = config_kuavo_env.control_mode
        self.obs_key_map = config_kuavo_env.obs_key_map
        self.only_arm = config_kuavo_env.only_arm
        self.eef_type = config_kuavo_env.eef_type
        self.which_arm = config_kuavo_env.which_arm
        self.qiangnao_dof_needed = config_kuavo_env.qiangnao_dof_needed

        self.is_binary = config_kuavo_env.is_binary
        self.head_init = config_kuavo_env.head_init
        self.arm_init = np.array([0]*14)


        # 从配置中获取limits部分
        self.limits = config_kuavo_env.limits
        self.obs_key_map = config_kuavo_env.obs_key_map
        self.obs_buffer = ObsBuffer(
            config=config_kuavo_env, 
            obs_key_map=self.obs_key_map,
        )
        self.arm_state_keys = config_kuavo_env.arm_state_keys # observation.state 的key顺序
        self.ratio = config_kuavo_env.ratio
        self.frame_alignment = config_kuavo_env.frame_alignment

    def _set_observation_space(self):
        limits = self.limits
        obs_low, obs_high = [], []

        # -------- 构建 state 空间（joint_q + gripper） --------
        if 'joint_q' in self.obs_key_map:
            joint_min, joint_max = limits['joint_q']['min'], limits['joint_q']['max']
        else:
            joint_min, joint_max = [], []
        if 'gripper' in self.obs_key_map:
            grip_min, grip_max = limits['gripper']['min'], limits['gripper']['max']
        else:
            grip_min, grip_max = [], []
        if self.which_arm == 'both':
            obs_low.extend(joint_min[:7]+grip_min[:1]+joint_min[7:14]+grip_min[1:2])
            obs_high.extend(joint_max[:7]+grip_max[:1]+joint_max[7:14]+grip_max[1:2])
        if self.which_arm == 'left':
            obs_low.extend(joint_min[:7]+grip_min[:1])
            obs_high.extend(joint_max[:7]+grip_max[:1])
        if self.which_arm == 'right':
            obs_low.extend(joint_min[7:14]+grip_min[1:2])
            obs_high.extend(joint_max[7:14]+grip_max[1:2])

        self.obs_low = np.array(obs_low)
        self.obs_high = np.array(obs_high)

        # -------- 构建图像空间 --------
        obs_spaces = {}
        for key, obs_name in self.obs_key_map.items():
            if any(tag in key for tag in ['cam', 'depth']):
                h, w = obs_name["handle"]["params"]["resize_wh"]
                if 'depth' in key:
                    low, high = obs_name['handle']['params']['depth_range']
                    obs_spaces[f"observation.{key}"] = gym.spaces.Box(
                        low=low, high=high, shape=(1, h, w), dtype=np.uint16
                    )
                else:  # cam 类键
                    obs_spaces[f"observation.images.{key}"] = gym.spaces.Box(
                        low=0, high=255, shape=(3, h, w), dtype=np.uint8
                    )

        # -------- 添加 state 空间 --------
        obs_spaces["observation.state"] = gym.spaces.Box(
            low=self.obs_low,
            high=self.obs_high,
            dtype=np.float32,
            shape=(len(self.obs_low),)
        )

        self.observation_space = gym.spaces.Dict(obs_spaces)

    def _set_action_space(self):
        limits = self.limits

        # ===============================
        # 辅助函数：构造单臂动作范围
        # ===============================
        def get_arm_action_range(arm: str):
            """返回 (low, high)"""
            if self.control_mode == 'joint':
                if arm == 'left':
                    return (
                        limits['joint_q']['min'][:7] + limits['gripper']['min'][:1],
                        limits['joint_q']['max'][:7] + limits['gripper']['max'][:1],
                    )
                elif arm == 'right':
                    return (
                        limits['joint_q']['min'][7:14] + limits['gripper']['min'][1:2],
                        limits['joint_q']['max'][7:14] + limits['gripper']['max'][1:2],
                    )
                elif arm == 'both':
                    return (
                        limits['joint_q']['min'][:7] + limits['gripper']['min'][:1]+limits['joint_q']['min'][7:14] + limits['gripper']['min'][1:2],
                        limits['joint_q']['max'][:7] + limits['gripper']['max'][:1]+limits['joint_q']['max'][7:14] + limits['gripper']['max'][1:2],
                    )

            elif self.control_mode == 'eef':
                # key = 'eef_relative' if self.use_delta else 'eef'
                key = 'eef'
                inf_pad = [-np.inf] * 6
                inf_pad_pos = [np.inf] * 6

                def eef_block(start, end, grip_idx):
                    low = limits[key]['min'][start:end] + inf_pad + limits['gripper']['min'][grip_idx:grip_idx + 1]
                    high = limits[key]['max'][start:end] + inf_pad_pos + limits['gripper']['max'][grip_idx:grip_idx + 1]
                    return low, high

                if arm == 'left':
                    return eef_block(0, 3, 0)
                elif arm == 'right':
                    return eef_block(6, 9, 1)
                elif arm == 'both':
                    low1, high1 = eef_block(0, 3, 0)
                    low2, high2 = eef_block(6, 9, 1)
                    return low1 + low2, high1 + high2

            raise ValueError(f"Unsupported arm mode: {arm}")

        # ===============================
        # 获取臂部动作范围
        # ===============================
        arm_low, arm_high = get_arm_action_range(self.which_arm)

        # ===============================
        # 如果包含 base 控制，则拼接 base 范围
        # ===============================
        if not self.only_arm:
            base_low, base_high = limits['base']['min'], limits['base']['max']
            arm_low += base_low
            arm_high += base_high

        # ===============================
        # 创建 Gym Box 空间
        # ===============================
        self.action_space = gym.spaces.Box(
            low=np.array(arm_low, dtype=np.float64),
            high=np.array(arm_high, dtype=np.float64),
            dtype=np.float64,
        )

    def _init_kuavo_sdk(self):
        """初始化Kuavo SDK"""
        if not KuavoSDK().Init():
            log_robot.error("Init KuavoSDK failed, exit!")
            sys.exit(1)
        self.robot = KuavoRobot()
        self.robot_state = KuavoRobotState()

    def _set_ros_topics(self):
        """设置ROS话题"""
        self.rate = rospy.Rate(self.ros_rate)
        
        if self.eef_type == 'rq2f85':
            self.pub_eef_joint = self.ros_manager.register_publisher('/gripper/command', JointState, queue_size=10)
        elif self.eef_type == 'leju_claw':
            self.lejuclaw = LejuClaw()
        elif self.eef_type == 'qiangnao':
            self.qiangnao = DexterousHand()
        # obs buffer 初始化            
        self.obs_buffer.wait_buffer_ready()


    def reset(self, **kwargs):
        """重置机器人状态"""
        self._enter_external_control_mode()
        self._reset_head()
        self._reset_eef()

        # === 平均当前观测和位姿 ===
        avg_data = self._compute_average_state(average_num=10)

        # === 更新状态 ===
        self.cur_state = avg_data["state"]
        self.cur_joint_angles_action = avg_data["joint_action"]

        obs = self.get_obs()
        self.sleep_time = 0
        self.average_sleep_time = 0
        return obs, {}

    # ==========================================================
    # 子函数 1. 外部控制模式设置
    # ==========================================================
    def _enter_external_control_mode(self):
        self.robot.set_external_control_arm_mode()
        print("set_external_control_arm_mode", self.robot_state.arm_control_mode())

    # ==========================================================
    # 子函数 2. 头部复位
    # ==========================================================
    def _reset_head(self):
        if self.head_init is not None:
            self.robot.control_head(self.head_init[0], self.head_init[1])

    # ==========================================================
    # 子函数 3. 末端执行器（夹爪）复位
    # ==========================================================
    def _reset_eef(self):
        if self.which_arm == 'both':
            if self.eef_type == 'qiangnao':
                self.qiangnao.control(target_positions=[0, 100, 0, 0, 0, 0, 0, 100, 0, 0, 0, 0], target_velocities=None, target_torques=None)
            elif self.eef_type == 'leju_claw':
                self.lejuclaw.control(target_positions=[0, 0], target_velocities=None, target_torques=None)
        elif self.which_arm == 'left':
            if self.eef_type == 'qiangnao':
                self.qiangnao.control_left(target_positions=[0, 100, 0, 0, 0, 0], target_velocities=None, target_torques=None)
            elif self.eef_type == 'leju_claw':
                self.lejuclaw.control_left(target_positions=[0], target_velocities=None, target_torques=None)
        elif self.which_arm == 'right':
            if self.eef_type == 'qiangnao':
                self.qiangnao.control_right(target_positions=[0, 100, 0, 0, 0, 0], target_velocities=None, target_torques=None)
            elif self.eef_type == 'leju_claw':
                self.lejuclaw.control_right(target_positions=[0], target_velocities=None, target_torques=None)
        else:
            raise KeyError(f"Unsupported arm type: {self.which_arm}")

    # ==========================================================
    # 子函数 4. 求平均状态与末端姿态
    # ==========================================================
    def _compute_average_state(self, average_num=10):
        state_sum, joint_sum = None, None

        for i in range(average_num):
            state = self.get_obs()
            fk_joint_angles = self._get_init_joint_angles(self.arm_state["joint_q"])

            if i == 0:
                state_sum = np.array(state["observation.state"], dtype=float)
                joint_sum = np.array(fk_joint_angles, dtype=float)
            else:
                state_sum += np.array(state["observation.state"], dtype=float)
                joint_sum += fk_joint_angles
            time.sleep(0.001)

        # 平均计算
        avg_state = state_sum / average_num
        avg_joint = joint_sum / average_num

        return {
            "state": avg_state,
            "joint_action": avg_joint,
        }

    # ==========================================================
    # 子函数 5. 构造完整 FK 输入
    # ==========================================================
    def _get_init_joint_angles(self, joint_q):
        # if self.which_arm == 'both':
        #     if self.fk_joint_angles_for_reset is not None:
        #         fk_joint_angles = np.array(self.fk_joint_angles_for_reset) / 180 * np.pi
        #     else:
        #         fk_joint_angles = np.array([-10, 15, 25, -85, -90, 15, -20,   50, 0, 0, -140, 90, 0, 0])/180*np.pi
        #     return fk_joint_angles
        if self.which_arm == 'both':
            return np.array(joint_q)
        elif self.which_arm == 'left':
            return np.concatenate((joint_q, self.arm_init[7:14]))
        elif self.which_arm == 'right':
            return np.concatenate((self.arm_init[:7], joint_q))
        else:
            raise ValueError(f"Invalid which_arm: {self.which_arm}")
    
    def check_action(self, action, mode='default'):
        if mode == 'default':  # 比较 action_space
            if len(action) != len(self.action_space.low):
                raise ValueError(f"action shape must be {len(self.action_space.low)}")
            if np.any(action < self.action_space.low) or np.any(action > self.action_space.high):
                log_robot.warning(
                    f"action out of range, action: {action}, "
                    f"low: {self.action_space.low}, high: {self.action_space.high}"
                )
                action = np.clip(action, self.action_space.low, self.action_space.high)
            return action

        raise ValueError(f"Unsupported mode: {mode}")



    def step(self, action):
        t0 = time.time()
        log_robot.info(f"action: {action}")
        # check clip action in action space
        action = self.check_action(action, mode='default')
        t1 = time.time()
        log_robot.info(f"clip action: {action}, check time: {t1 - t0:.3f}s")

        if not self.only_arm:
            # 获取action中base移动相关的部分（最后4个值），最后一位用于判断是移动还是手部动作
            base_action = action[-4:]
            move_flag = base_action[-1]  # 0-1之间的值，用于判断是否执行base移动
            if move_flag > 0.5:  # 如果大于0.5，执行base移动
                # 执行base移动，这里需要调用相关的base移动接口
                log_robot.info(f"➡️  执行【底盘移动】(mode_flag > 0.5)")
                log_robot.info(f"cmd_pos_world = [x:{base_action[0]:.4f}, y:{base_action[1]:.4f}, yaw:{base_action[2]:.4f}], move_flag={move_flag:.4f}")
                self.robot.control_command_pose_world(base_action[0], base_action[1], 0, base_action[2])
                self.rate.sleep()
                self._record_sleep_time(t1)
                return self.get_obs(), 0, False, False, {}
            # 如果不执行base移动，则执行手部动作，此时使用action的前面部分
            action = action[:-4]


        # === 4. 执行动作 ===
        t2 = time.time()
        self.cur_joint_angles_action = np.concatenate((action[:7], action[8:15]), axis=0)
        self.exec_action(action)

        # === 5. 延时与观测 ===
        
        self.rate.sleep()
        self._record_sleep_time(t2)
        t3 = time.time()
        obs = self.get_obs()
        t4 = time.time()
        log_robot.info(f"get obs time: {t4 - t3:.3f}s")

        # === 6. 奖励与返回 ===
        reward = self.compute_reward()
        return obs, reward, False, False, {}
    
    
    def _record_sleep_time(self, t_start):
        self.sleep_time = time.time() - t_start
        self.average_sleep_time += self.sleep_time
        log_robot.info(f"rate.sleep time: {self.sleep_time:.3f}s")

    def exec_action(self, action):
        """执行机械臂与末端执行器动作"""
        # if not self.only_arm:
        #     return

        def safe_control_arm(target_position):
            try:
                self.robot.control_arm_joint_positions(target_position)
            except RuntimeError as e:
                # 当机器人处于 command_pose_world 状态（底盘移动）时，无法控制手臂
                if "must be in stance state" in str(e):
                    log_robot.warning(f"⚠️  无法发送手臂命令：机器人当前状态不允许 (可能正在底盘移动)")
                    log_robot.debug(f"   详细错误: {e}")
                else:
                    raise


        if self.which_arm == 'both':
            left_joints, left_eef = action[:7], action[7]
            right_joints, right_eef = action[8:15], action[15]
            target_position = np.concatenate((left_joints, right_joints), axis=0)
            safe_control_arm(target_position)
            self._control_eef(left_eef, right_eef)

        elif self.which_arm == 'left':
            left_joints, left_eef = action[:7], action[7]
            target_position = np.concatenate((left_joints, self.arm_init[7:14]), axis=0)
            safe_control_arm(target_position)
            self._control_eef(left_eef, 0)

        elif self.which_arm == 'right':
            right_joints, right_eef = action[:7], action[7]
            target_position = np.concatenate((self.arm_init[:7], right_joints), axis=0)
            safe_control_arm(target_position)
            self._control_eef(0, right_eef)
        else:
            raise KeyError(f"Unsupported which_arm: {self.which_arm}")



    def _control_eef(self, left_eef, right_eef):
        """根据 eef_type 控制不同的末端执行器"""
        if self.eef_type == 'rq2f85':
            eef_msg = JointState()
            try:
                eef_msg.name = ['left_gripper_joint', 'right_gripper_joint']
            except Exception as e:
                log_robot.info(f"_control_eef error! {e}")
            eef_msg.position = np.array([left_eef * 255, right_eef * 255])
            self.pub_eef_joint.publish(eef_msg)

        elif self.eef_type == 'leju_claw':
            eef_msg = JointState()
            eef_msg.position = np.array([left_eef * 100, right_eef * 100])
            self.lejuclaw.control(target_positions=eef_msg.position)

        elif self.eef_type == 'qiangnao':
            if self.qiangnao_dof_needed != 1:
                raise KeyError("qiangnao_dof_needed != 1 is not supported!")

            tem_left, tem_right = left_eef * 100, right_eef * 100
            target_positions = np.array([
                tem_left, 100, *([tem_left] * 4),
                tem_right, 100, *([tem_right] * 4)
            ])
            self.qiangnao.control(target_positions=target_positions)

        else:
            raise KeyError(f"Unsupported eef_type: {self.eef_type}")

    def compute_reward(self):
        """计算奖励"""
        return 0

    def get_obs(self):
        """获取观测图像及state等"""
        obs = {}
        self.arm_state = {}

        if self.frame_alignment:
            obs_from_buffer = self.obs_buffer.get_aligned_obs(reference_keys=None, max_dt=1/self.ros_rate,ratio=self.ratio)
            if obs_from_buffer is None or not all(v is not None for v in obs_from_buffer.values()):
                obs_from_buffer = self.obs_buffer.get_aligned_obs(reference_keys=None, max_dt=float('inf'),ratio=self.ratio)
        else:
            obs_from_buffer = self.obs_buffer.get_latest_obs()
        
        for k,v in obs_from_buffer.items():
            # remap key
            if 'depth' in k:
                obs[f"observation.{k}"] = v
            elif 'cam' in k:
                obs[f"observation.images.{k}"] = v
            else:
                self.arm_state[f"{k}"] = v

        if self.is_binary:
            self.arm_state['gripper'] = np.where(self.arm_state['gripper']>0.5, 1, 0)

        assert len(self.arm_state.keys()) >= 2, f"arm_state must have exactly 2 elements, but got {len(self.arm_state.keys())}"

        state_keys = [k for k in self.arm_state_keys if k in self.arm_state]

        arm_data = { "left": [], "right": [] }

        for key in state_keys:
            data = self.arm_state[key]
            if len(data) == 0:
                continue
            mid = len(data) // 2
            if self.which_arm == "both":
                arm_data["left"].append(data[:mid])
                arm_data["right"].append(data[mid:])
            elif self.which_arm == "left":
                arm_data["left"].append(data)
            elif self.which_arm == "right":
                arm_data["right"].append(data)
            else:
                raise KeyError(f"Unsupported which_arm: {self.which_arm}")

        # 拼接结果
        obs["observation.state"] = np.concatenate(
            arm_data["left"] + arm_data["right"], axis=0
        )
        log_robot.info(f"STATE: contained {state_keys}, concated value: {obs['observation.state']}")

        obs["observation.state"] = torch.from_numpy(obs["observation.state"]).float().unsqueeze(0)
        return obs    

    def close(self):
        """关闭环境，释放资源"""
        log_robot.info("Closing KuavoBaseRosEnv...")
        try:
            if hasattr(self, 'obs_buffer'):
                self.obs_buffer.stop_subscribers()
                if hasattr(self.obs_buffer, 'obs_buffer_data'):
                    for k in self.obs_buffer.obs_buffer_data:
                        self.obs_buffer.obs_buffer_data[k]["data"].clear()
                        self.obs_buffer.obs_buffer_data[k]["timestamp"].clear()
                if hasattr(self.obs_buffer, 'ros_manager'):
                    self.obs_buffer.ros_manager = None
                if hasattr(self.obs_buffer, 'control_signal_manager'):
                    self.obs_buffer.control_signal_manager = None
                del self.obs_buffer
            
            if hasattr(self, 'ros_manager'):
                self.ros_manager.close()
            if hasattr(self, 'control_signal_manager'):
                self.control_signal_manager.close()
            log_robot.info("KuavoBaseRosEnv closed successfully.")
        except Exception as e:
            log_robot.error(f"Error closing KuavoBaseRosEnv: {e}")
            traceback.print_exc()

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()

class LejuClaw:
    """乐聚爪手控制器"""
    def __init__(self, ros_manager=None):
        self.ros_manager = ros_manager or ROSManager()
        self._pub_leju_claw_cmd = self.ros_manager.register_publisher('/leju_claw_command', lejuClawCommand, queue_size=10)

    def control(self, target_positions: list, target_velocities: list = None, target_torques: list = None):
        """控制双手"""
        self._validate_inputs(target_positions, target_velocities, target_torques, 2)
        
        cmd = lejuClawCommand()
        cmd.data.name = ['left_claw', 'right_claw']
        
        target_positions = [max(0.0, min(100.0, pos)) for pos in target_positions]
        target_velocities = self._get_default_velocities(target_velocities, 2)
        target_torques = self._get_default_torques(target_torques, 2)
        
        cmd.data.position = target_positions
        cmd.data.velocity = target_velocities
        cmd.data.effort = target_torques
        self._pub_leju_claw_cmd.publish(cmd)

    def control_left(self, target_positions: list, target_velocities: list = None, target_torques: list = None):
        """控制左手"""
        self._validate_inputs(target_positions, target_velocities, target_torques, 1)
        self.control(
            [target_positions[0], 0],
            [target_velocities[0] if target_velocities else 90, 0],
            [target_torques[0] if target_torques else 1.0, 0]
        )

    def control_right(self, target_positions: list, target_velocities: list = None, target_torques: list = None):
        """控制右手"""
        self._validate_inputs(target_positions, target_velocities, target_torques, 1)
        self.control(
            [0, target_positions[0]],
            [0, target_velocities[0] if target_velocities else 90],
            [0, target_torques[0] if target_torques else 1.0]
        )

    def _validate_inputs(self, positions, velocities, torques, expected_len):
        """验证输入参数"""
        assert len(positions) == expected_len, f"target_positions must be a list of length {expected_len}"
        if velocities is not None:
            assert len(velocities) == expected_len, f"target_velocities must be a list of length {expected_len}"
        if torques is not None:
            assert len(torques) == expected_len, f"target_torques must be a list of length {expected_len}"

    def _get_default_velocities(self, velocities, length):
        """获取默认速度"""
        if velocities is None:
            return [90] * length
        return [max(0.0, min(100.0, vel)) for vel in velocities]

    def _get_default_torques(self, torques, length):
        """获取默认力矩"""
        if torques is None:
            return [1.0] * length
        return [max(0.0, min(10.0, torque)) for torque in torques]

    def close(self):
        """释放资源"""
        if hasattr(self, 'ros_manager'):
            self.ros_manager.close()

# 使用示例
if __name__ == "__main__":
    from kuavo_deploy.config import load_kuavo_config
    
    # 使用上下文管理器确保资源正确释放
    with KuavoBaseRosEnv(load_kuavo_config()) as env:
        obs, info = env.reset()

        for _ in range(1):
            obs = env.get_obs()
            env.rate.sleep()
            print(obs.keys())
            for k, v in obs.items():
                print(k, v.shape)
                print(v.max(), v.min())
