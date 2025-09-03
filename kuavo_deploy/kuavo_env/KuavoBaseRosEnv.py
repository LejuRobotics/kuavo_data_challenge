#!/usr/bin/env python3
import threading
import rospy
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Float32, Float32MultiArray,Float64MultiArray,Int32
from std_msgs.msg import Bool
import cv2
import gymnasium as gym
import time
from configs.deploy.config_kuavo_env import load_kuavo_env_config
import sys
from kuavo_humanoid_sdk import KuavoSDK,KuavoRobot,KuavoRobotState,LejuClaw,DexterousHand
from kuavo_deploy.utils.logging_utils import setup_logger

import traceback
# import Rq2f85ClawCmd,Rq2f85ClawState

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

class KuavoBaseRosEnv(gym.Env):

    def __init__(self, config_path: str = None):
        log_robot.info(f"config_path: {config_path}")
        config_kuavo_env = load_kuavo_env_config(config_path)
        self.real = config_kuavo_env.real
        self.ros_rate = config_kuavo_env.ros_rate
        self.control_mode = config_kuavo_env.control_mode
        self.image_size = config_kuavo_env.image_size
        self.only_arm = config_kuavo_env.only_arm
        self.eef_type = config_kuavo_env.eef_type
        self.which_arm = config_kuavo_env.which_arm
        self.qiangnao_dof_needed = config_kuavo_env.qiangnao_dof_needed
        self.leju_claw_dof_needed = config_kuavo_env.leju_claw_dof_needed
        self.rq2f85_dof_needed = config_kuavo_env.rq2f85_dof_needed
        self.arm_init = config_kuavo_env.arm_init
        self.slice_robot = config_kuavo_env.slice_robot
        self.qiangnao_slice = config_kuavo_env.qiangnao_slice
        self.claw_slice = config_kuavo_env.claw_slice
        self.is_binary = config_kuavo_env.is_binary
        self.head_init = config_kuavo_env.head_init

        self.arm_min = config_kuavo_env.arm_min
        self.arm_max = config_kuavo_env.arm_max
        self.arm_min = np.array(self.arm_min)/180*np.pi
        self.arm_max = np.array(self.arm_max)/180*np.pi
        self.eef_min = config_kuavo_env.eef_min
        self.eef_max = config_kuavo_env.eef_max

        self.input_images = config_kuavo_env.input_images

        self.bridge = CvBridge()

        if self.only_arm:
            # Initialize action space based on control mode
            if self.control_mode == 'joint':
                if self.which_arm == 'both':
                    self.arm_joint_dim = 14
                    action_low = np.concatenate((self.arm_min[:7], self.eef_min, self.arm_min[7:14], self.eef_min), axis=0)
                    action_high = np.concatenate((self.arm_max[:7], self.eef_max, self.arm_max[7:14], self.eef_max), axis=0)
                    state_low = np.concatenate((self.arm_min[:7], self.eef_min, self.arm_min[7:14], self.eef_min), axis=0)
                    state_high = np.concatenate((self.arm_max[:7], self.eef_max, self.arm_max[7:14], self.eef_max), axis=0)
                elif self.which_arm == 'left':
                    self.arm_joint_dim = 7
                    action_low = np.concatenate((self.arm_min[:7], self.eef_min), axis=0)
                    action_high = np.concatenate((self.arm_max[:7], self.eef_max), axis=0)
                    state_low = np.concatenate((self.arm_min[:7], self.eef_min), axis=0)
                    state_high = np.concatenate((self.arm_max[:7], self.eef_max), axis=0)
                elif self.which_arm == 'right':
                    self.arm_joint_dim = 7
                    action_low = np.concatenate((self.arm_min[7:], self.eef_min), axis=0)
                    action_high = np.concatenate((self.arm_max[7:], self.eef_max), axis=0)
                    state_low = np.concatenate((self.arm_min[7:], self.eef_min), axis=0)
                    state_high = np.concatenate((self.arm_max[7:], self.eef_max), axis=0)
            elif self.control_mode == 'eef':
                raise KeyError("control_mode = 'eef' is not supported!")
                # self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(14,))

            self.action_space = gym.spaces.Box(low=action_low, high=action_high, shape=(len(action_low),), dtype=np.float64)


            self.observation_space = gym.spaces.Dict({
                        "observation.state": gym.spaces.Box(low=state_low, high=state_high, shape=(len(state_low),)),
                        })
            height,width = self.image_size
            for image in self.input_images:
                if 'depth' in image:
                    self.observation_space[f"observation.{image}"] = gym.spaces.Box(0, 65535, shape=(1,height,width), dtype=np.uint16)
                else:
                    self.observation_space[f"observation.images.{image}"] = gym.spaces.Box(0, 255, shape=(height,width,3), dtype=np.uint8)

            # Initialize state variables
            self.head_cam_h_img = None
            self.wrist_cam_l_img = None
            self.wrist_cam_r_img = None
            self.state = None
            self.start_state = None
        else:
            raise KeyError("only_arm = False is not supported!")
        
        self.init_kuavo_sdk()
        self.initial_topics()
        while not self.check_rostopics():
            if not check_control_signals():
                log_robot.info("🛑 收到停止信号，退出机械臂运动")
                sys.exit(1)
            log_robot.info(f"Waiting for inializing...")
            time.sleep(1)
        log_robot.info(f"Inializing done!")

    # 推荐的初始化方式
    def safe_init_node(self):
        try:
            rospy.init_node('kuavo_base_ros_env', anonymous=True)
            log_robot.info("Node initialized successfully!")
            return True
        except rospy.ROSException as e:
            log_robot.error(f"Node initialization failed: {e}")
            return False

    def init_kuavo_sdk(self):
        if not KuavoSDK().Init():  # Init! !!! IMPORTANT !!!
            print("Init KuavoSDK failed, exit!")
            sys.exit(1)
        self.robot = KuavoRobot()
        self.robot_state = KuavoRobotState()

    def initial_topics(self):
        
        self.rate = rospy.Rate(self.ros_rate)
        
        if self.only_arm:
            # ROS subscribers
            rospy.Subscriber("/cam_h/color/image_raw/compressed", CompressedImage, self.cam_h_callback)
            rospy.Subscriber("/cam_l/color/image_raw/compressed", CompressedImage, self.cam_l_callback)
            rospy.Subscriber("/cam_r/color/image_raw/compressed", CompressedImage, self.cam_r_callback)
            rospy.Subscriber("/cam_h/depth/image_raw/compressedDepth", CompressedImage, self.cam_h_depth_callback)
            rospy.Subscriber("/cam_l/depth/image_rect_raw/compressedDepth", CompressedImage, self.cam_l_depth_callback)
            rospy.Subscriber("/cam_r/depth/image_rect_raw/compressedDepth", CompressedImage, self.cam_r_depth_callback)
            
            if not self.real:
                rospy.Subscriber("/gripper/state", JointState, self.gripper_state_callback)
                # rospy.Subscriber("/F_state", JointState, self.F_state_callback)

                # ROS publishers
                if self.eef_type == 'rq2f85':
                    self.pub_eef_joint = rospy.Publisher('/gripper/command', JointState, queue_size=10)
                elif self.eef_type == 'leju_claw':
                    self.pub_eef_joint = rospy.Publisher('/claw_cmd', JointState, queue_size=10)
                elif self.eef_type == 'qiangnao':
                    raise KeyError("qiangnao is not supported!")
            else:
                # ROS subscribers
                
                if self.eef_type == 'leju_claw':
                    self.lejuclaw = LejuClaw()
                elif self.eef_type == 'qiangnao':
                    self.qiangnao = DexterousHand()
        
    def compute_reward(self):
        return 0

    def check_rostopics(self):
        """
        检查ROS话题可用性
        """
        
        # 根据配置确定需要检查的话题
        topics = {
            "/sensors_data_raw": "kuavo_msgs/sensorsData",
        }
        
        if self.only_arm:
            topics.update({
                "/cam_h/color/image_raw/compressed": "CompressedImage",
                "/cam_l/color/image_raw/compressed": "CompressedImage",
                "/cam_r/color/image_raw/compressed": "CompressedImage",
                "/cam_h/depth/image_raw/compressedDepth": "CompressedImage",
                "/cam_l/depth/image_rect_raw/compressedDepth": "CompressedImage", 
                "/cam_r/depth/image_rect_raw/compressedDepth": "CompressedImage"
            })
            
            if not self.real:
                if self.eef_type == 'rq2f85':
                    topics["/gripper/state"] = "sensor_msgs/JointState"
                elif self.eef_type == 'leju_claw':
                    topics["/leju_claw_state"] = "kuavo_msgs/lejuClawState"
                elif self.eef_type == 'qiangnao':
                    topics["/dexhand/state"] = "sensor_msgs/JointState"
            else:
                if self.eef_type == 'leju_claw':
                    topics["/leju_claw_state"] = "kuavo_msgs/lejuClawState"
                elif self.eef_type == 'qiangnao':
                    topics["/dexhand/state"] = "sensor_msgs/JointState"
        
        log_robot.info(f"检查ROS话题 ({len(topics)}个):")
        log_robot.info("=" * 50)
        
        available = 0
        for topic, msg_type in topics.items():
            try:
                # 动态导入消息类型
                if msg_type == "CompressedImage":
                    from sensor_msgs.msg import CompressedImage
                    msg_class = CompressedImage
                elif msg_type == "sensor_msgs/JointState":
                    from sensor_msgs.msg import JointState
                    msg_class = JointState
                elif msg_type == "kuavo_msgs/sensorsData":
                    from kuavo_humanoid_sdk.msg.kuavo_msgs.msg import sensorsData
                    msg_class = sensorsData
                elif msg_type == "kuavo_msgs/lejuClawState":
                    from kuavo_humanoid_sdk.msg.kuavo_msgs.msg import lejuClawState
                    msg_class = lejuClawState
                else:
                    from std_msgs.msg import AnyMsg
                    msg_class = AnyMsg
                
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

    def reset(self, **kwargs):
        # change arm control mode to external control
        self.robot.set_external_control_arm_mode()
        print("set_external_control_arm_mode",self.robot_state.arm_control_mode())

        # reset head
        if self.head_init is not None:
            self.robot.control_head(self.head_init[0], self.head_init[1])

        if self.real:
            if self.which_arm == 'both':
                if self.eef_type == 'qiangnao':
                    target_positions = [0,100,0,0,0,0,0,100,0,0,0,0]
                    self.qiangnao.control(target_positions=target_positions, target_velocities=None, target_torques=None)
                elif self.eef_type == 'leju_claw':
                    raise KeyError("leju_claw is not supported!")
            elif self.which_arm == 'left':
                if self.eef_type == 'qiangnao':
                    target_positions = [0,100,0,0,0,0]
                    self.qiangnao.control_left(target_positions=target_positions, target_velocities=None, target_torques=None)
                elif self.eef_type == 'leju_claw':
                    raise KeyError("leju_claw is not supported!")
            elif self.which_arm == 'right':
                if self.eef_type == 'qiangnao':
                    target_positions = [0,100,0,0,0,0]
                    self.qiangnao.control_right(target_positions=target_positions, target_velocities=None, target_torques=None)
                elif self.eef_type == 'leju_claw':
                    raise KeyError("leju_claw is not supported!")
            else:
                raise KeyError("which_arm != 'left' or 'right' or 'both' is not supported!")

        average_num = 10
        for i in range(average_num):
            state = self.get_obs()
            if i==0:
                self.start_state = state["observation.state"]
            else:
                self.start_state += state["observation.state"]
            time.sleep(0.001)
        self.start_state = self.start_state / average_num

        obs = self.get_obs()
        return obs, {}

    def step(self, action):
        # Execute action
        log_robot.info(f"action: {action}")
        # arm action in rad, eef action in 0-1
        action = np.clip(action, self.action_space.low, self.action_space.high) # 限制动作范围 
        log_robot.info(f"clip action: {action}")
        self.exec_action(action)
        self.rate.sleep()
        
        # Get new observation
        obs = self.get_obs()
        
        # Simplified reward and termination
        reward = self.compute_reward()
        done = False
        return obs, reward, done, False, {}

    def exec_action(self, action):
        if self.only_arm:
            if not self.real:
                if self.which_arm == 'both':
                    if self.control_mode == 'eef':
                        raise KeyError("control_mode = 'eef' is not supported!")
                    elif self.control_mode == 'joint':
                        target_position = np.concatenate((action[:7], action[8:15]), axis=0)
                        self.robot.control_arm_joint_positions(target_position)
                    if self.eef_type == 'rq2f85':
                        if self.rq2f85_dof_needed == 1:   
                            eef_msg = JointState()
                            eef_msg.name = ['left_gripper_joint','right_gripper_joint']
                            eef_msg.position = np.concatenate(([action[7]*255], [action[15]*255]), axis=0)
                            # log_robot.info("eef_msg.position", eef_msg.position)
                            self.pub_eef_joint.publish(eef_msg)
                        else:
                            raise KeyError("rq2f85_dof_needed != 1 is not supported!")
                    elif self.eef_type == 'leju_claw':
                        if self.leju_claw_dof_needed == 1:                        
                            eef_msg = JointState()
                            eef_msg.position = np.concatenate(([action[7]*100], [action[15]*100]), axis=0)
                            self.pub_eef_joint.publish(eef_msg)
                        else:
                            raise KeyError("leju_claw_dof_needed != 1 is not supported!")
                    elif self.eef_type == 'qiangnao':
                        raise KeyError("qiangnao is not supported!")
                elif self.which_arm == 'left':
                    if self.control_mode == 'eef':
                        raise KeyError("control_mode = 'eef' is not supported!")
                    elif self.control_mode == 'joint':
                        target_position = np.concatenate((action[:7], self.arm_init[7:14]), axis=0)
                        self.robot.control_arm_joint_positions(target_position)
                    if self.eef_type == 'rq2f85':
                        if self.rq2f85_dof_needed == 1:   
                            eef_msg = JointState()
                            eef_msg.name = ['left_gripper_joint','right_gripper_joint']
                            eef_msg.position = np.concatenate(([action[7]*255], [0]), axis=0)
                            self.pub_eef_joint.publish(eef_msg)
                        else:
                            raise KeyError("rq2f85_dof_needed != 1 is not supported!")
                    elif self.eef_type == 'leju_claw':
                        if self.leju_claw_dof_needed == 1:
                            eef_msg = JointState()
                            eef_msg.position = np.concatenate(([action[7]*100], [0]), axis=0)
                            self.pub_eef_joint.publish(eef_msg)
                        else:
                            raise KeyError("leju_claw_dof_needed != 1 is not supported!")
                    elif self.eef_type == 'qiangnao':
                        raise KeyError("qiangnao is not supported!")
                elif self.which_arm == 'right':
                    if self.control_mode == 'eef':
                        raise KeyError("control_mode = 'eef' is not supported!")
                    elif self.control_mode == 'joint':
                        target_position = np.concatenate((self.arm_init[:7],action[:7]), axis=0)
                        self.robot.control_arm_joint_positions(target_position)
                    if self.eef_type == 'rq2f85':
                        if self.rq2f85_dof_needed == 1:   
                            eef_msg = JointState()
                            eef_msg.name = ['left_gripper_joint','right_gripper_joint']
                            eef_msg.position = np.concatenate(([0], [action[15]*255]), axis=0)
                            self.pub_eef_joint.publish(eef_msg)
                        else:
                            raise KeyError("rq2f85_dof_needed != 1 is not supported!")
                    elif self.eef_type == 'leju_claw':
                        if self.leju_claw_dof_needed == 1:
                            eef_msg = JointState()
                            eef_msg.position = np.concatenate(([0],[action[7]*100]), axis=0)
                            self.pub_eef_joint.publish(eef_msg)
                        else:
                            raise KeyError("leju_claw_dof_needed != 1 is not supported!")
                    elif self.eef_type == 'qiangnao':
                        raise KeyError("qiangnao is not supported!")
                else:
                    raise KeyError("which_arm != 'left' or 'right' or 'both' is not supported!")
            else:
                if self.which_arm == 'both':
                    if self.control_mode == 'eef':
                        raise KeyError("control_mode = 'eef' is not supported!")
                    elif self.control_mode == 'joint':
                        target_positions = np.concatenate((action[:7], action[8:15]), axis=0)
                        self.robot.control_arm_joint_positions(target_positions)
                    if self.eef_type == 'leju_claw':
                        if self.leju_claw_dof_needed == 1:
                            target_positions = np.concatenate(([action[7]*100], [action[15]*100]), axis=0)
                            self.lejuclaw.control(target_positions=target_positions)
                        else:
                            raise KeyError("leju_claw_dof_needed != 1 is not supported!")
                    elif self.eef_type == 'qiangnao':
                        if self.qiangnao_dof_needed == 1:
                            tem_left = action[7]*100
                            tem_right = action[15]*100
                            target_positions = np.concatenate(([tem_left], [100], [tem_left]*4,[tem_right], [100], [tem_right]*4), axis=0)
                            self.qiangnao.control(target_positions=target_positions)
                        else:
                            raise KeyError("qiangnao_dof_needed != 1 is not supported!")
                elif self.which_arm == 'left':
                    if self.control_mode == 'eef':
                        raise KeyError("control_mode = 'eef' is not supported!")
                    elif self.control_mode == 'joint':
                        target_positions = np.concatenate((action[:7], self.arm_init[7:14]), axis=0)
                        self.robot.control_arm_joint_positions(target_positions)
                    if self.eef_type == 'leju_claw':
                        if self.leju_claw_dof_needed == 1:
                            target_positions = np.concatenate(([action[7]*100], [0]), axis=0)
                            self.lejuclaw.control(target_positions=target_positions)
                        else:
                            raise KeyError("leju_claw_dof_needed != 1 is not supported!")
                    elif self.eef_type == 'qiangnao':
                        if self.qiangnao_dof_needed == 1:
                            tem_left = action[7]*100
                            tem_right = 0
                            target_positions = np.concatenate(([tem_left], [100], [tem_left]*4,[tem_right], [100], [tem_right]*4), axis=0)
                            self.qiangnao.control(target_positions=target_positions)
                        else:
                            raise KeyError("qiangnao_dof_needed != 1 is not supported!")
                elif self.which_arm == 'right':
                    if self.control_mode == 'eef':
                        raise KeyError("control_mode = 'eef' is not supported!")
                    elif self.control_mode == 'joint':
                        target_positions = np.concatenate((self.arm_init[:7],action[:7]), axis=0)
                        self.robot.control_arm_joint_positions(target_positions)
                    if self.eef_type == 'leju_claw':
                        if self.leju_claw_dof_needed == 1:
                            target_positions = np.concatenate(([0],[action[7]*100]), axis=0)
                            self.lejuclaw.control(target_positions=target_positions)
                        else:
                            raise KeyError("leju_claw_dof_needed != 1 is not supported!")
                    elif self.eef_type == 'qiangnao':
                        if self.qiangnao_dof_needed == 1:
                            tem_left = 0
                            tem_right = action[7]*100
                            target_positions = np.concatenate(([tem_left], [100], [tem_left]*4,[tem_right], [100], [tem_right]*4), axis=0)
                            self.qiangnao.control(target_positions=target_positions)
                        else:
                            raise KeyError("qiangnao_dof_needed != 1 is not supported!")
                else:
                    raise KeyError("which_arm != 'left' or 'right' or 'both' is not supported!")
        else:
            raise KeyError("only_arm = False is not supported!")

    def get_obs(self):
        self.joint_q = self.robot_state.arm_joint_state().position
        # self.joint_q = np.array(self.joint_q) + np.random.normal(0, 0.001, len(self.joint_q))
        if self.real:
            self.eef_state = np.concatenate((self.robot_state.eef_state()[0].position, self.robot_state.eef_state()[1].position), axis=0)
            if self.is_binary:
                self.eef_state = np.where(self.eef_state>50, 1, 0)
            else:
                self.eef_state = self.eef_state/100
        else:
            self.eef_state = np.array(self.gripper_state)
            if self.is_binary:
                self.eef_state = np.where(self.eef_state>0.4, 1, 0)
            else:
                self.eef_state = self.eef_state/0.8

        if self.only_arm:
            if self.which_arm == 'both':
                if self.eef_type == 'rq2f85':
                    self.state = np.concatenate((self.joint_q[:7], self.eef_state[:self.rq2f85_dof_needed],self.joint_q[7:],self.eef_state[-self.rq2f85_dof_needed:]), axis=0)
                elif self.eef_type == 'qiangnao':
                    self.state = np.concatenate((self.joint_q[:7], self.eef_state[:self.qiangnao_dof_needed],self.joint_q[7:],self.eef_state[-self.qiangnao_dof_needed:]), axis=0)
                elif self.eef_type == 'leju_claw':
                    self.state = np.concatenate((self.joint_q[:7], self.eef_state[:self.leju_claw_dof_needed],self.joint_q[7:],self.eef_state[-self.leju_claw_dof_needed:]), axis=0)
            elif self.which_arm == 'left':
                if self.eef_type == 'rq2f85':
                    self.state = np.concatenate((self.joint_q[:7], self.eef_state[:self.rq2f85_dof_needed]), axis=0)
                elif self.eef_type == 'qiangnao':
                    self.state = np.concatenate((self.joint_q[:7], self.eef_state[:self.qiangnao_dof_needed]), axis=0)
                elif self.eef_type == 'leju_claw':
                    self.state = np.concatenate((self.joint_q[:7], self.eef_state[:self.leju_claw_dof_needed]), axis=0)
            elif self.which_arm == 'right':
                if self.eef_type == 'rq2f85':
                    self.state = np.concatenate((self.joint_q[7:], self.eef_state[-self.rq2f85_dof_needed:]), axis=0)
                elif self.eef_type == 'qiangnao':
                    self.state = np.concatenate((self.joint_q[7:], self.eef_state[-self.qiangnao_dof_needed:]), axis=0)
                elif self.eef_type == 'leju_claw':
                    self.state = np.concatenate((self.joint_q[7:], self.eef_state[-self.leju_claw_dof_needed:]), axis=0)
            else:
                raise KeyError("which_arm != 'left' or 'right' or 'both' is not supported!")

        obs = {"observation.state": self.state}
        for image in self.input_images:
            if 'depth' in image:
                obs[f"observation.{image}"] = getattr(self, f"{image}_img")
            else:
                obs[f"observation.images.{image}"] = getattr(self, f"{image}_img")
        return obs

    def process_rgb_img(self, msg):
        # 处理 CompressedImage
        img_arr = np.frombuffer(msg.data, dtype=np.uint8)
        # print("img_arr.max",img_arr.max())
        cv_img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        if cv_img is None:
            raise ValueError("Failed to decode compressed image")
        # 色域转换由BGR->RGB
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        height,width = self.image_size
        img_array = np.array(cv_img)
        # check image size
        if img_array.shape[0] != height or img_array.shape[1] != width:
            raise ValueError(f"Image size is not correct! {img_array.shape} != {height}x{width}")
        return img_array

    # ROS callbacks
    def cam_h_callback(self, msg):
        self.head_cam_h_img = self.process_rgb_img(msg)

    def cam_l_callback(self, msg):
        self.wrist_cam_l_img = self.process_rgb_img(msg)
    
    def cam_r_callback(self, msg):
        # print("cam_r_callback!")
        self.wrist_cam_r_img = self.process_rgb_img(msg)

    def process_depth_img(self, msg):
        if not (hasattr(msg, 'format') and hasattr(msg, 'data')):
            raise ValueError(f"Skipping invalid message")

        # print(f"message format: {msg.format}")

        png_magic = bytes([137, 80, 78, 71, 13, 10, 26, 10])
        idx = msg.data.find(png_magic)
        if idx == -1:
            raise ValueError("PNG header not found, unable to decode.")

        png_data = msg.data[idx:]
        np_arr = np.frombuffer(png_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
        if image is None:
            print("cv2.imdecode also failed")
            return None

        if image.dtype != np.uint16:
            print("Warning: The decoded image is not a 16-bit image, actual dtype: ", image.dtype)
        # check image size
        height,width = self.image_size
        if image.shape[0] != height or image.shape[1] != width:
            raise ValueError(f"Depth image size is not correct! {image.shape} != {height}x{width}")
        # print("depth image dtype: ", depth_image.dtype)
        return image[np.newaxis,...]

    def cam_h_depth_callback(self, msg):
        self.depth_h_img = self.process_depth_img(msg)
    
    def cam_l_depth_callback(self, msg):
        self.depth_l_img = self.process_depth_img(msg)
    
    def cam_r_depth_callback(self, msg):
        self.depth_r_img = self.process_depth_img(msg)

    def gripper_state_callback(self, msg):
        self.gripper_state = msg.position

    def F_state_callback(self, msg): # Used in simulation
        all_joint_angle = msg.position
        if self.only_arm:
            joint = all_joint_angle[:28]
            if self.which_arm == 'both':
                if self.eef_type == 'leju_claw':
                    claw = all_joint_angle[28:]
                    output_state = joint[12:19]
                    if self.leju_claw_dof_needed == 1:
                        output_state = np.insert(output_state, 7, claw[0])
                        output_state = np.concatenate((output_state, joint[19:26]), axis=0)
                        output_state = np.insert(output_state, 15, claw[8])
                    else:
                        raise KeyError("leju_claw_dof_needed != 1 is not supported!")
                elif self.eef_type == 'qiangnao':
                    raise KeyError("qiangnao is not supported!")
            elif self.which_arm == 'left':
                if self.eef_type == 'leju_claw':
                    claw = all_joint_angle[28:]
                    output_state = joint[12:19]
                    if self.leju_claw_dof_needed == 1:
                        output_state = np.insert(output_state, 7, claw[0])
                    else:
                        raise KeyError("leju_claw_dof_needed != 1 is not supported!")
                elif self.eef_type == 'qiangnao':
                    raise KeyError("qiangnao is not supported!")
            elif self.which_arm == 'right':
                if self.eef_type == 'leju_claw':
                    claw = all_joint_angle[28:]
                    output_state = joint[19:26]
                    if self.leju_claw_dof_needed == 1:
                        output_state = np.insert(output_state, 7, claw[8])
                    else:
                        raise KeyError("leju_claw_dof_needed != 1 is not supported!")
                elif self.eef_type == 'qiangnao':
                    raise KeyError("qiangnao is not supported!")
        else:
            raise KeyError("only_arm = False is not supported!")
        
        self.state = output_state

# Usage example
if __name__ == "__main__":
    env = KuavoBaseRosEnv(config_path="configs/deploy/kuavo_sim_env.yaml")
    obs, info = env.reset()

    for _ in range(1):
        obs = env.get_obs()
        env.rate.sleep()
        print(obs.keys())
        for k,v in obs.items():
            print(k,v.shape)
            print(v.max(),v.min())
        # print(obs["observation.state"])
    
    # for _ in range(100):
    #     action = [0]*16
    #     obs, reward, done, _, info = env.step(action)
    #     print(obs.keys())
    #     if done:
    #         obs, info = env.reset()