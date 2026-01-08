#!/usr/bin/env python3
# pip install --extra-index-url https://rospypi.github.io/simple/ rospy rosbag
# pip install roslz4 --extra-index-url https://rospypi.github.io/simple/
import numpy as np
import cv2
import rosbag
from pprint import pprint
import os
import glob
from collections import defaultdict
from typing import Callable, Optional

# ================ 机器人关节信息定义 ================

DEFAULT_LEG_JOINT_NAMES=[
    "l_leg_roll", "l_leg_yaw", "l_leg_pitch", "l_knee", "l_foot_pitch", "l_foot_roll",
    "r_leg_roll", "r_leg_yaw", "r_leg_pitch", "r_knee", "r_foot_pitch", "r_foot_roll",
]
DEFAULT_ARM_JOINT_NAMES = [
    "zarm_l1_link", "zarm_l2_link", "zarm_l3_link", "zarm_l4_link", "zarm_l5_link", "zarm_l6_link", "zarm_l7_link",
    "zarm_r1_link", "zarm_r2_link", "zarm_r3_link", "zarm_r4_link", "zarm_r5_link", "zarm_r6_link", "zarm_r7_link",
]
DEFAULT_HEAD_JOINT_NAMES = [
    "head_yaw", "head_pitch"
]
DEFAULT_DEXHAND_JOINT_NAMES = [
    "left_qiangnao_1", "left_qiangnao_2","left_qiangnao_3","left_qiangnao_4","left_qiangnao_5","left_qiangnao_6",
    "right_qiangnao_1", "right_qiangnao_2","right_qiangnao_3","right_qiangnao_4","right_qiangnao_5","right_qiangnao_6",
]
DEFAULT_LEJUCLAW_JOINT_NAMES = [
    "left_claw", "right_claw",
]

DEFAULT_JOINT_NAMES_LIST = DEFAULT_LEG_JOINT_NAMES + DEFAULT_ARM_JOINT_NAMES + DEFAULT_HEAD_JOINT_NAMES

DEFAULT_JOINT_NAMES = {
    "full_joint_names": DEFAULT_LEG_JOINT_NAMES + DEFAULT_ARM_JOINT_NAMES + DEFAULT_HEAD_JOINT_NAMES,
    "leg_joint_names": DEFAULT_LEG_JOINT_NAMES,
    "arm_joint_names": DEFAULT_ARM_JOINT_NAMES,
    "head_joint_names": DEFAULT_HEAD_JOINT_NAMES,
}



# ================ 数据转换信息定义 ================
def init_parameters(cfg):

    global DEFAULT_CAMERA_NAMES, TRAIN_HZ, MAIN_TIMELINE_FPS, SAMPLE_DROP, CONTROL_HAND_SIDE
    global SLICE_ROBOT, SLICE_DEX, SLICE_CLAW
    global IS_BINARY, DELTA_ACTION, RELATIVE_START
    global RESIZE_W, RESIZE_H
    global ONLY_HALF_UP_BODY, USE_LEJU_CLAW, USE_QIANGNAO
    global USE_DEPTH, DEPTH_RANGE
    global TASK_DESCRIPTION

    
    from .config_dataset import load_config
    config = load_config(cfg)

    # 从配置文件加载基本设置
    USE_DEPTH = config.use_depth
    DEPTH_RANGE = config.depth_range
    DEFAULT_CAMERA_NAMES = config.default_camera_names
    TRAIN_HZ = config.train_hz
    MAIN_TIMELINE_FPS = config.main_timeline_fps
    SAMPLE_DROP = config.sample_drop
    CONTROL_HAND_SIDE = config.which_arm

    # 根据which_arm自动计算的切片配置
    SLICE_ROBOT = config.slice_robot
    SLICE_DEX = config.dex_slice
    SLICE_CLAW = config.claw_slice

    # 处理标志
    IS_BINARY = config.is_binary
    DELTA_ACTION = config.delta_action
    RELATIVE_START = config.relative_start

    # 图像尺寸设置
    RESIZE_W = config.resize.width
    RESIZE_H = config.resize.height

    # 根据配置自动推导的标志
    ONLY_HALF_UP_BODY = config.only_half_up_body  # 总是True当only_arm为True时
    USE_LEJU_CLAW = config.use_leju_claw  # 由eef_type决定
    USE_QIANGNAO = config.use_qiangnao  # 由eef_type决定

    TASK_DESCRIPTION = config.task_description  # 任务描述


# ================ 数据处理函数定义 ==================
class KuavoMsgProcesser:
    """
    Kuavo 话题处理函数
    """
    @staticmethod
    def process_color_image(msg):
        """
        Process the color image.
        Args:
            msg (sensor_msgs.msg.Image or sensor_msgs.msg.CompressedImage): The color image message.
        Returns:
             Dict:
                - data(np.ndarray): Image data with shape (height, width, 3).
                - "timestamp" (float): The timestamp of the image.
        """
        if hasattr(msg, 'encoding'):
            if msg.encoding != 'rgb8':
                # Handle different encodings here if necessary
                raise ValueError(f"Unsupported encoding: {msg.encoding}. Expected 'rgb8'.")

            # Convert the ROS Image message to a numpy array
            img_arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)

            # If the image is in 'bgr8' format, convert it to 'rgb8'
            if msg.encoding == 'bgr8':
                cv_img = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
            else:
                cv_img = img_arr
        else:
            # 处理 CompressedImage
            img_arr = np.frombuffer(msg.data, dtype=np.uint8)
            cv_img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            if cv_img is None:
                raise ValueError("Failed to decode compressed image")
            # 色域转换由BGR->RGB
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        cv_img=cv2.resize(cv_img,(RESIZE_W,RESIZE_H)) ### ATT: resize the image to 640x480(w * h)
        return {"data": cv_img, "timestamp": msg.header.stamp.to_sec()}

    @staticmethod
    def process_depth_image(msg):
        if not (hasattr(msg, 'format') and hasattr(msg, 'data')):
            print(f"Skipping invalid message")

        # print(f"message format: {msg.format}")

        png_magic = bytes([137, 80, 78, 71, 13, 10, 26, 10])
        idx = msg.data.find(png_magic)
        if idx == -1:
            print("PNG header not found, unable to decode.")
            return None

        png_data = msg.data[idx:]
        np_arr = np.frombuffer(png_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
        if image is None:
            print("cv2.imdecode also failed")
            return None

        if image.dtype != np.uint16:
            print("Warning: The decoded image is not a 16-bit image, actual dtype: ", image.dtype)
        depth_image = cv2.resize(image, (RESIZE_W, RESIZE_H), interpolation=cv2.INTER_NEAREST)
        # print("depth image dtype: ", depth_image.dtype)
        # return {"data": depth_image[np.newaxis,...], "timestamp": msg.header.stamp.to_sec()}
        return {"data": depth_image, "timestamp": msg.header.stamp.to_sec()}


    @staticmethod
    def process_joint_state(msg):
        """
            Args:
                msg (kuavo_msgs/sensorsData): The joint state message.
            Returns:
                Dict:
                    - data(np.ndarray): The joint state data with shape (28,).
                    - "timestamp" (float): The timestamp of the joint state.
        """
        # radian
        joint_q = msg.joint_data.joint_q
        return {"data": joint_q, "timestamp": msg.header.stamp.to_sec()}

    @staticmethod
    def process_joint_cmd(msg):
        """
            Args:
                msg (kuavo_msgs/jointCmd): The joint state message.

            Returns:
                Dict:
                    - data(np.ndarray): The joint state data with shape (28,).
                    - "timestamp" (float): The timestamp of the joint state.
        """
        # radian
        return {"data": msg.joint_q, "timestamp": msg.header.stamp.to_sec()}
    
    @staticmethod
    def process_kuavo_arm_traj(msg):
        """Process the arm trajectory command.

        Args:
            msg (sensor_msgs/JointState): The arm trajectory command message.

        Returns:
            Dict: A dictionary containing the processed arm trajectory data.
        """
        
        # radian
        return {"data": np.deg2rad(msg.position), "timestamp": msg.header.stamp.to_sec()}
    
    @staticmethod
    def process_cmd_pos_world(msg):    #cmd_pose_world处理函数
        linear = [msg.twist.linear.x, msg.twist.linear.y]
        angular = [msg.twist.angular.z]
        
        twist_data = linear + angular
        
        return {"data": twist_data, "timestamp": msg.header.stamp.to_sec()}

    @staticmethod
    def process_claw_state(msg):
        """
            Args:
                msg (kuavo_sdk/lejuClawState): The claw state message.
            Returns:
                Dict:
                    - data(np.ndarray): The claw state data with shape (2,).
                    - "state" (float): The state of the claws state.
        """
        state= msg.data.position
        return { "data": state, "timestamp": msg.header.stamp.to_sec() }
    @staticmethod
    def process_claw_cmd(msg):
        position= msg.data.position
        return { "data": position, "timestamp": msg.header.stamp.to_sec() }
    
    @staticmethod
    def process_rq2f85_cmd(msg):
        # position= list([msg.position])
        # position.extend(list([msg.right_cmd]))
        position = msg.position
        return { "data": position, "timestamp": msg.header.stamp.to_sec() }

    @staticmethod
    def process_rq2f85_state(msg):
        # position= list([msg.left_position])
        # position.extend(list([msg.right_position]))
        position = msg.position
        return { "data": position, "timestamp": msg.header.stamp.to_sec() }
    
    @staticmethod
    def process_qiangnao_state(msg):
        state= list(msg.left_hand_position)
        state.extend(list(msg.right_hand_position))
        return { "data": state, "timestamp": msg.header.stamp.to_sec() }
    
    @staticmethod
    def process_dex_state(msg):
        return { "data": msg.position, "timestamp": msg.header.stamp.to_sec() }

    @staticmethod
    def process_qiangnao_cmd(msg):
        position= list(msg.left_hand_position)
        position.extend(list(msg.right_hand_position))
        return { "data": position, "timestamp": msg.header.stamp.to_sec() }
    

    @staticmethod
    def process_sensors_data_raw_extract_imu(msg):
        imu_data = msg.imu_data
        gyro = imu_data.gyro
        acc = imu_data.acc
        free_acc = imu_data.free_acc
        quat = imu_data.quat

        # 将数据合并为一个NumPy数组
        imu = np.array([gyro.x, gyro.y, gyro.z,
                        acc.x, acc.y, acc.z,
                        free_acc.x, free_acc.y, free_acc.z,
                        quat.x, quat.y, quat.z, quat.w])

        return {"data": imu, "timestamp": msg.header.stamp.to_sec()}

    @staticmethod
    def process_sensors_data_raw_extract_arm(msg):
        """
        Processes raw joint state data from a given message by extracting the portion relevant to the arm.

        Parameters:
            msg: The input message containing joint state information.

        Returns:
            dict: A dictionary with processed joint state data. The 'data' field is sliced to include only indices 12 through 25.

        Notes:
            This function uses KuavoMsgProcesser.process_joint_state to initially process the input message and then extracts the specific range of data for further use.
        """
        res = KuavoMsgProcesser.process_joint_state(msg)
        res["data"] = res["data"][12:26]
        return res

    @staticmethod
    def process_joint_cmd_extract_arm(msg):
        res = KuavoMsgProcesser.process_joint_cmd(msg)
        res["data"] = res["data"][12:26]
        return res

    @staticmethod
    def process_sensors_data_raw_extract_arm_head(msg):
        res = KuavoMsgProcesser.process_joint_state(msg)
        res["data"] = res["data"][12:]
        return res

    @staticmethod
    def process_joint_cmd_extract_arm_head(msg):
        res = KuavoMsgProcesser.process_joint_cmd(msg)
        res["data"] = res["data"][12:]
        return res


class KuavoRosbagReader:
    def __init__(self):
        self._msg_processer = KuavoMsgProcesser()
        self._topic_process_map = {
            "observation.state": {
                "topic": "/sensors_data_raw",
                "msg_process_fn": self._msg_processer.process_joint_state,
            },
            "action.kuavo_arm_traj": {
                "topic": "/kuavo_arm_traj",
                "msg_process_fn": self._msg_processer.process_kuavo_arm_traj,
            },
            # 连续版kauvo_arm_traj,用于task4
            "action.kuavo_arm_traj_alt": {
                "topic": "/kuavo_arm_traj_synced",
                "msg_process_fn": self._msg_processer.process_kuavo_arm_traj,
            },
            # 新action：cmd_pos_world，控制位置
            "action.cmd_pos_world": {
                "topic": "/cmd_pose_world_synced",
                "msg_process_fn": self._msg_processer.process_cmd_pos_world,
            },

            "action": {
                "topic": "/joint_cmd",
                "msg_process_fn": self._msg_processer.process_joint_cmd,
            },
            "observation.imu": {
                "topic": "/sensors_data_raw",
                "msg_process_fn": self._msg_processer.process_sensors_data_raw_extract_imu,
            },
            "observation.claw": {
                # 末端数据二指夹爪的位置状态信息
                "topic": "/leju_claw_state",
                "msg_process_fn": self._msg_processer.process_claw_state,
            },
            "action.claw": {
                # 末端数据二指夹爪的运动信息位置
                "topic": "/leju_claw_command",
                "msg_process_fn": self._msg_processer.process_claw_cmd,
            },
            "observation.qiangnao": {
                "topic": "/dexhand/state",
                "msg_process_fn": self._msg_processer.process_dex_state,
            },
            "action.qiangnao": {
                "topic": "/control_robot_hand_position",
                "msg_process_fn": self._msg_processer.process_qiangnao_cmd,
            },
            "observation.rq2f85": {
                "topic": "/gripper/state",
                "msg_process_fn": self._msg_processer.process_rq2f85_state,
            },
            "action.rq2f85": {
                "topic": "/gripper/command",
                "msg_process_fn": self._msg_processer.process_rq2f85_cmd,
            },
        }
        for camera in DEFAULT_CAMERA_NAMES:
            # observation.images.{camera}.depth  => color images
            # if 'wrist' in camera or 'head_cam_h' in camera:
            #     self._topic_process_map[f"{camera}"] = {
            #         "topic": f"/{camera[-5:]}/color/image_raw/compressed",   # "/{camera}/color/compressed", 新刷的20.04orin镜像可以直接发布压缩图像，不用额外的压缩节点
            #         "msg_process_fn": self._msg_processer.process_color_image,
            #     }
            if 'wrist_cam_l' in camera:
                self._topic_process_map[f"{camera}"] = {
                    "topic": "/cam_l/color/image_raw/compressed",   ### ATT: 这里的cam_r是因为在2025年7月8日的rosbag中，cam_r是左手腕相机
                    "msg_process_fn": self._msg_processer.process_color_image,
                }
            elif 'wrist_cam_r' in camera:
                self._topic_process_map[f"{camera}"] = {
                    "topic": "/cam_r/color/image_raw/compressed",
                    "msg_process_fn": self._msg_processer.process_color_image,
                }
            elif 'head_cam_h' in camera:
                self._topic_process_map[f"{camera}"] = {
                    "topic": "/cam_h/color/image_raw/compressed",
                    "msg_process_fn": self._msg_processer.process_color_image,
            }
            elif 'head_cam_l' in camera:
                self._topic_process_map[f"{camera}"] = {
                "topic": f"/zedm/zed_node/left/image_rect_color/compressed",
                "msg_process_fn": self._msg_processer.process_color_image,
            }
            elif 'head_cam_r' in camera:
                self._topic_process_map[f"{camera}"] = {
                "topic": f"/zedm/zed_node/right/image_rect_color/compressed",
                "msg_process_fn": self._msg_processer.process_color_image,
            }
            elif "depth_h" in camera:
                self._topic_process_map[f"{camera}"] = {
                "topic": f"/cam_h/depth/image_raw/compressedDepth",
                "msg_process_fn": self._msg_processer.process_depth_image, 
                }
            elif "depth_l" in camera:
                self._topic_process_map[f"{camera}"] = {
                "topic": f"/cam_l/depth/image_rect_raw/compressedDepth",
                "msg_process_fn": self._msg_processer.process_depth_image, 
                }
            elif "depth_r" in camera:
                self._topic_process_map[f"{camera}"] = {
                "topic": f"/cam_r/depth/image_rect_raw/compressedDepth",
                "msg_process_fn": self._msg_processer.process_depth_image, 
                }



    def load_raw_rosbag(self, bag_file: str):
        try:
            bag = rosbag.Bag(bag_file)      
            return bag
        except rosbag.bag.ROSBagUnindexedException:
            print(f"Bag file {bag_file} is unindexed, attempting to reindex...")
            from common.utils import reindex_rosbag
            reindexed_file = reindex_rosbag(bag_file)
            if reindexed_file:
                try:
                    bag = rosbag.Bag(reindexed_file)
                    return bag
                except Exception as e:
                    print(f"Error loading reindexed bag file: {e}")
                    raise RuntimeError(f"Failed to load reindexed bag file: {reindexed_file}")
            else:
                # 尝试允许未索引的方式打开
                print(f"Reindexing failed, trying to open with allow_unindexed=True")
                try:
                    bag = rosbag.Bag(bag_file, 'r', allow_unindexed=True)
                    print(f"Successfully opened unindexed bag: {bag_file}")
                    return bag
                except Exception as e:
                    print(f"Failed to open unindexed bag: {e}")
                    raise RuntimeError(f"Failed to reindex and load bag file: {bag_file}")
        except Exception as e:
            print(f"Error loading bag file {bag_file}: {e}")
            raise
    
    def print_bag_info(self, bag: rosbag.Bag):
        pprint(bag.get_type_and_topic_info().topics)
     
    def process_rosbag(self, bag_file: str):
        """
        Process the rosbag file and return the processed data.
        
        Note: This method loads all data into memory. For large rosbags, 
        consider using process_rosbag_streaming() or process_rosbag_chunked() instead.

        Args:
            bag_file (str): The path to the rosbag file.

        Returns:
            Dict: The processed data.
        """
        bag = self.load_raw_rosbag(bag_file)
        data = {}
        for key, topic_info in self._topic_process_map.items():
            topic = topic_info["topic"]
            msg_process_fn = topic_info["msg_process_fn"]
            data[key] = []
            for _, msg, t in bag.read_messages(topics=topic):
                msg_data = msg_process_fn(msg)
                # 如果没有 header.stamp或者时间戳是远古时间不合要求，使用bag的时间戳 
                correct_timestamp = t.to_sec() 
                msg_data["timestamp"] = correct_timestamp
                data[key].append(msg_data)
        
        data_aligned = self.align_frame_data(data)
        
        return data_aligned
    
    
    def process_rosbag_chunked(
        self,
        bag_file: str,
        frame_callback: Callable[[dict, int], None],
        chunk_size: int = 100,
        save_callback: Optional[Callable[[], None]] = None
    ) -> int:
        """
        分块流式处理rosbag（推荐用于超大rosbag）
        
        参考Diffusion Policy的按需读取方式：
        1. 第一遍扫描：只读取时间戳（内存占用极小，只有几MB）
        2. 第二遍扫描：按时间窗口分块读取+对齐+处理
        
        与process_rosbag的区别：
        - process_rosbag: 一次性加载所有数据到内存（内存峰值巨大）
        - process_rosbag_chunked: 分块读取，边读边对齐边处理（内存可控）
        
        Args:
            bag_file: rosbag文件路径
            frame_callback: 处理每帧的回调函数 (aligned_frame, frame_idx) -> None
                           aligned_frame包含所有话题的对齐数据
            chunk_size: 每个chunk包含的帧数（默认100帧）
            save_callback: 每个chunk处理完后的回调（用于保存dataset和释放内存）
        
        Returns:
            处理的总帧数
            
        Example:
            def on_frame(aligned_frame, frame_idx):
                # 处理对齐后的帧，添加到dataset
                dataset.add_frame(...)
            
            def on_chunk_done():
                # 保存当前chunk，释放内存
                dataset.save_episode()
                gc.collect()
            
            reader.process_rosbag_chunked(
                bag_file="large.bag",
                frame_callback=on_frame,
                chunk_size=100,
                save_callback=on_chunk_done
            )
        """
        from kuavo_data.common.chunk_process import ChunkedRosbagProcessor
        
        processor = ChunkedRosbagProcessor(
            msg_processer=self._msg_processer,
            topic_process_map=self._topic_process_map,
            camera_names=DEFAULT_CAMERA_NAMES,
            train_hz=TRAIN_HZ,
            main_timeline_fps=MAIN_TIMELINE_FPS,
            sample_drop=SAMPLE_DROP,
            only_half_up_body=ONLY_HALF_UP_BODY
        )
        
        # 第一遍：只扫描时间戳（内存占用极小）
        main_timeline, main_timestamps, all_timestamps = processor.scan_timestamps_only(bag_file)
        
        # 第二遍：分块处理
        return processor.process_in_chunks(
            bag_file=bag_file,
            main_timestamps=main_timestamps,
            all_timestamps=all_timestamps,
            frame_callback=frame_callback,
            chunk_size=chunk_size,
            save_callback=save_callback
        )
    
    def align_frame_data(self, data: dict):
        aligned_data = defaultdict(list)
        main_timeline = max(
            DEFAULT_CAMERA_NAMES,
            key=lambda cam_k: len(data.get(cam_k, [])),
        )

        jump = MAIN_TIMELINE_FPS // TRAIN_HZ

        # 注意：当 SAMPLE_DROP 为 0 时，不能使用 [SAMPLE_DROP:-SAMPLE_DROP]，
        # 因为 [-0] 等价于 [0]，会导致切片为空列表。
        main_timeline_list = [t['timestamp'] for t in data[main_timeline]]
        if SAMPLE_DROP > 0:
            main_img_timestamps = main_timeline_list[SAMPLE_DROP:-SAMPLE_DROP][::jump]
        else:
            main_img_timestamps = main_timeline_list[::jump]
        min_end = min([data[k][-1]['timestamp'] for k in data.keys() if len(data[k]) > 0])
        main_img_timestamps = [t for t in main_img_timestamps if t < min_end]
        
        # 特殊处理kuavo_arm_traj话题的时间戳连续性检测（可能有断点，需要999处理）在控制下肢时候
        def detect_timestamp_gaps(timestamps, gap_threshold=0.15 * 10 / TRAIN_HZ): # gap_threshold可调
            """检测时间戳中的间隙，返回间隙位置"""
            if len(timestamps) < 2:
                return []
            
            gaps = []
            for i in range(1, len(timestamps)):
                if timestamps[i] - timestamps[i-1] > gap_threshold:
                    gaps.append(i)
            return gaps
        
        gaps = []
        # 检查kuavo_arm_traj是否有断点，如果有则进行特殊处理
        if not ONLY_HALF_UP_BODY and "action.kuavo_arm_traj" in data and len(data["action.kuavo_arm_traj"]) > 0:
            arm_traj_timestamps = [t['timestamp'] for t in data["action.kuavo_arm_traj"]]
            gaps = detect_timestamp_gaps(arm_traj_timestamps)
            
            # 只有在检测到间隙时才进行特殊处理
            if len(gaps) > 0:
                print(f"Detected {len(gaps)} gaps in action.kuavo_arm_traj, applying 999 flag processing")
                
                # 获取数据维度
                data_dim = len(data["action.kuavo_arm_traj"][0]['data'])
                
                # 创建完整的时间序列，在间隙处填充999
                complete_arm_traj_data = []
                
                for stamp in main_img_timestamps:
                    # 检查当前时间戳是否在间隙中
                    in_gap = False
                    for gap_idx in gaps:
                        if gap_idx < len(arm_traj_timestamps) and arm_traj_timestamps[gap_idx] > stamp:
                            # 检查是否在间隙范围内
                            if gap_idx > 0 and arm_traj_timestamps[gap_idx-1] < stamp < arm_traj_timestamps[gap_idx]:
                                in_gap = True
                                break
                    
                    if in_gap:
                        # 在间隙中，使用999填充
                        flag_data = np.full(data_dim, 999.0, dtype=np.float32)
                        complete_arm_traj_data.append({
                            "data": flag_data,
                            "timestamp": stamp
                        })
                        print(f"Created flag data (999) for action.kuavo_arm_traj at timestamp {stamp}")
                    else:
                        # 正常时间戳，使用最近的数据
                        time_array = np.array(arm_traj_timestamps)
                        idx = np.argmin(np.abs(time_array - stamp))
                        complete_arm_traj_data.append(data["action.kuavo_arm_traj"][idx])
                
                # 将处理后的数据添加到aligned_data
                aligned_data["action.kuavo_arm_traj"] = complete_arm_traj_data
                print(f"Processed action.kuavo_arm_traj with gap detection: {len(complete_arm_traj_data)} frames, {len(gaps)} gaps detected")
            else:
                print("No gaps detected in action.kuavo_arm_traj, using normal alignment")
                # 没有间隙，使用正常对齐逻辑（在下面的循环中处理）
        
        # 处理其他话题的正常对齐
        for stamp in main_img_timestamps:
            stamp_sec = stamp
            for key, v in data.items():
                # 跳过已经特殊处理的kuavo_arm_traj（只有在有间隙时才跳过）
                if key == "action.kuavo_arm_traj" and len(gaps) > 0:
                    continue
                
                if len(v) > 0:
                    this_obs_time_seq = [this_frame['timestamp'] for this_frame in v]
                    time_array = np.array([t for t in this_obs_time_seq])
                    idx = np.argmin(np.abs(time_array - stamp_sec))
                    aligned_data[key].append(v[idx])
                else:
                    aligned_data[key] = []
        
        # 打印对齐结果（安全地处理空数据情况）
        if len(main_img_timestamps) > 0 and len(aligned_data) > 0:
            # 使用main_timeline作为参考，或者使用aligned_data中的第一个非空键
            reference_key = main_timeline if main_timeline in aligned_data and len(aligned_data[main_timeline]) > 0 else next(iter([k for k, v in aligned_data.items() if len(v) > 0]), None)
            if reference_key is not None:
                original_len = len(data.get(main_timeline, []))
                aligned_len = len(aligned_data[reference_key])
                print(f"Aligned {reference_key}: {original_len} -> {aligned_len}")
            else:
                print("Warning: No aligned data found")
        else:
            print(f"Warning: No timestamps to align (main_img_timestamps={len(main_img_timestamps)}, aligned_data keys={list(aligned_data.keys())})")
        
        # 打印每个键的对齐结果
        for k, v in aligned_data.items():
            if len(v) > 0:
                if len(v) >= 2:
                    print(v[0]['timestamp'], v[1]['timestamp'], "length", k, len(v))
                else:
                    print(v[0]['timestamp'], "length", k, len(v))
        
        return aligned_data
    
    
    def list_bag_files(self, bag_dir: str):
        bag_files = glob.glob(os.path.join(bag_dir, '*.bag'))
        bag_files.sort()
        return bag_files
    
    def process_rosbag_dir(self, bag_dir: str):
        all_data = []
        # 按照文件名排序，获取 bag 文件列表
        bag_files = self.list_bag_files(bag_dir)
        episode_id = 0
        for bf in bag_files:
            print(f"Processing bag file: {bf}")
            episode_data = self.process_rosbag(bf)
            all_data.append(episode_data)
        
        return all_data
    
    

if __name__ == '__main__':
    bag_file = '/Users/wason/Code/RobotEmbodiedData/lerobot/data/testcamera/00001/testcamera_20250213_193331.bag'
    bag_dir = '/Users/wason/Code/RobotEmbodiedData/lerobot/data/testcamera2/'
    
    bag_file_1 = '/home/leju-ali/hx/kuavo/Task12_zed_dualArm/rosbag/rosbag_2025-03-15-15-21-40.bag'

    
    reader = KuavoRosbagReader()
    
    # reader.process_rosbag_dir(bag_dir)
    
    data_raw = reader.process_rosbag(bag_file_1)
    # data_aligned = reader.align_frame_data(data_raw)
    
    # print(data.keys())