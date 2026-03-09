"""
分块流式处理rosbag模块

核心思路（参考Diffusion Policy的按需读取方式）：
1. 第一遍扫描：只读取时间戳，确定主时间线（不加载图像数据，内存占用极小）
2. 第二遍扫描：按时间窗口分块读取，边读取边对齐边写入dataset

与原始方法的区别：
- 原始：一次性加载所有数据到内存 → 对齐 → 写入dataset（内存峰值巨大）
- 新方法：分块读取 → 即时对齐 → 即时写入 → 释放内存（内存占用可控）
"""

import numpy as np
import rosbag
from collections import defaultdict
from typing import Dict, List, Callable, Optional, Tuple, Generator
import logging
import bisect

logger = logging.getLogger(__name__)


class ChunkedRosbagProcessor:
    """
    分块流式处理rosbag，实现边读取边对齐边处理
    
    工作流程：
    1. scan_timestamps(): 第一遍扫描，只读取时间戳（内存占用极小）
    2. process_chunks(): 第二遍扫描，按时间窗口分块处理
    """
    
    def __init__(self, msg_processer, topic_process_map: dict, 
                 camera_names: list, train_hz: int, main_timeline: str, main_timeline_fps: int, 
                 sample_drop: int, only_half_up_body: bool):
        self._msg_processer = msg_processer
        self._topic_process_map = topic_process_map
        self.camera_names = camera_names
        self.train_hz = train_hz
        self.main_timeline_fps = main_timeline_fps
        self.sample_drop = sample_drop
        self.only_half_up_body = only_half_up_body
        self.main_timeline = main_timeline
    
    def scan_timestamps_only(self, bag_file: str) -> Tuple[str, List[float], Dict[str, List[float]]]:
        """
        第一遍扫描：只读取时间戳，不加载数据
        
        内存占用：只有时间戳列表（几MB），不包含图像数据
        
        Returns:
            main_timeline: 主时间线话题key
            main_timestamps: 对齐后的主时间戳序列（降采样后）
            all_timestamps: 每个话题的原始时间戳列表
        """
        bag = self._load_bag(bag_file)
        
        # 统计每个话题的时间戳（不加载消息内容）
        all_timestamps = defaultdict(list)
        topic_to_key = {}
        for k, v in self._topic_process_map.items():
            if v["topic"] not in topic_to_key.keys():
                topic_to_key[v["topic"]] = [k]
            else:
                topic_to_key[v["topic"]].append(k)
        
        logger.info(f"[Phase 1] Scanning timestamps from {bag_file}...")
        
        # 一次遍历，收集所有话题的时间戳
        all_topics = [v["topic"] for v in self._topic_process_map.values()]
        for topic, msg, t in bag.read_messages(topics=all_topics):
            keys = topic_to_key.get(topic)
            for key in keys:
                all_timestamps[key].append(t.to_sec())
                # all_timestamps[key].append(msg.header.stamp.to_sec())
                # all_timestamps[key].append(msg.header.stamp.to_sec() if hasattr(msg, 'header') and hasattr(msg.header, 'stamp') else t.to_sec())
        
        bag.close()                           
        
        # 确定主时间线：消息最多的相机
        camera_counts = {k: len(all_timestamps.get(k, [])) for k in self.camera_names}
        if not any(camera_counts.values()):
            raise ValueError("No camera data found in rosbag, please check your camera topics or bag file.")
        if self.main_timeline is None:
            main_timeline = max(camera_counts, key=lambda k: camera_counts[k])
        else:
            main_timeline = self.main_timeline
        logger.info(f"Main timeline: {main_timeline} ({camera_counts[main_timeline]} frames)")
        
        # 生成对齐后的主时间戳序列
        jump = self.main_timeline_fps // self.train_hz
        raw_timestamps = all_timestamps[main_timeline]

        if len(raw_timestamps) < 2 * self.sample_drop + 1:
            raise ValueError(f"Not enough frames: {len(raw_timestamps)}")

        # 丢弃首尾帧，降采样
        # 注意：当 sample_drop 为 0 时，不能使用 [self.sample_drop:-self.sample_drop]，
        # 因为 [-0] 等价于 [0]，会导致切片结果为空。
        if self.sample_drop > 0:
            main_timestamps = raw_timestamps[self.sample_drop:-self.sample_drop][::jump]
        else:
            main_timestamps = raw_timestamps[::jump]
        
        max_end = max(
            ts_list[-1]
            for ts_list in all_timestamps.values()
            if len(ts_list) > 0
        )

        # 裁剪主时间线，只保留所有话题都还存在数据的时间点
        before_len = len(main_timestamps)
        main_timestamps = [t for t in main_timestamps if t < max_end]
        after_len = len(main_timestamps)

        logger.info(
            f"Trim main timeline by max_end={max_end:.6f}, "
            f"frames: {before_len} -> {after_len}"
        )
        
        logger.info(f"Generated {len(main_timestamps)} aligned timestamps "
                   f"(from {len(raw_timestamps)} raw frames, "
                   f"dropped {self.sample_drop} frames at each end, jump={jump})")
        logger.info(f"Main timeline time range: [{main_timestamps[0]:.3f}, {main_timestamps[-1]:.3f}]")
        return main_timeline, main_timestamps, dict(all_timestamps)
    
    def process_in_chunks(
        self,
        bag_file: str,
        main_timestamps: List[float],
        all_timestamps: Dict[str, List[float]],
        frame_callback: Callable[[dict, int], None],
        chunk_size: int = 100,
        save_callback: Optional[Callable[[], None]] = None
    ) -> int:
        """
        第二遍扫描：按时间窗口分块处理
        
        策略：
        1. 将main_timestamps分成多个chunk
        2. 对于每个chunk，只读取该时间范围内的消息
        3. 对齐后立即调用frame_callback
        4. 每个chunk处理完后调用save_callback释放内存
        
        Args:
            bag_file: rosbag文件路径
            main_timestamps: 对齐后的主时间戳序列
            all_timestamps: 每个话题的原始时间戳列表（用于快速查找）
            frame_callback: 处理每帧的回调函数 (aligned_frame, frame_idx) -> None
            chunk_size: 每个chunk包含的帧数
            save_callback: 每个chunk处理完后的回调（用于保存和释放内存）
        
        Returns:
            处理的总帧数
        """
        bag = self._load_bag(bag_file)
        
        # 1. 识别夹爪相关的 topic keys (带有 action 且包含 claw/qiangnao/rqf85/gripper 等字眼)
        gripper_keys = []
        for k in self._topic_process_map.keys():
            k_lower = k.lower()
            if 'action' in k_lower and any(kw in k_lower for kw in ['claw', 'qiangnao', 'rq2f85', 'gripper']):
                gripper_keys.append(k)
                
        # 2. 维护一个跨 chunk 的状态，主要针对夹爪保持先前状态
        last_known_state = {}
        
        # 3. 如果开始就没有夹爪数据，需要用 00 替代。
        # 这里预先从 bag 中抽读一条以获取正确的 shape 填充 0，保证一开始就有合法的初始状态。
        if gripper_keys:
            logger.info(f"Prefetching initial shapes for gripper topics: {gripper_keys}")
            for key in gripper_keys:
                topic = self._topic_process_map[key]["topic"]
                try:
                    for _, msg, t in bag.read_messages(topics=[topic]):
                        msg_process_fn = self._topic_process_map[key]["msg_process_fn"]
                        msg_data = msg_process_fn(msg)
                        if "data" in msg_data:
                            # 拷贝格式并将 data 用全 0 初始化
                            last_known_state[key] = msg_data.copy()
                            last_known_state[key]["data"] = np.zeros_like(msg_data["data"])
                            last_known_state[key]["timestamp"] = 0.0
                        break  # 只读第一条用于获取shape
                except Exception as e:
                    logger.warning(f"Failed to prefetch initial state for {key}: {e}")

        # 为每个话题构建时间戳索引（用于快速查找最近帧）
        timestamp_arrays = {k: np.array(v) for k, v in all_timestamps.items()}
        
        # 预计算每个主时间戳对应的各话题索引（避免重复查找）
        alignment_indices = self._precompute_alignment_indices(
            main_timestamps, timestamp_arrays
        )
        
        # 检测kuavo_arm_traj的时间戳间隙
        arm_traj_gaps = self._detect_arm_traj_gaps(all_timestamps)
        
        num_chunks = (len(main_timestamps) + chunk_size - 1) // chunk_size
        total_frames = 0
        
        logger.info(f"[Phase 2] Processing {len(main_timestamps)} frames in {num_chunks} chunks...")
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, len(main_timestamps))
            chunk_timestamps = main_timestamps[start_idx:end_idx]
            
            if not chunk_timestamps:
                continue
            
            # 确定该chunk的时间范围（扩展一点以确保对齐数据可用）
            time_margin = 1.0 / self.train_hz  # 一帧的时间
            chunk_start_time = chunk_timestamps[0] - time_margin
            chunk_end_time = chunk_timestamps[-1] + time_margin
            
            logger.debug(f"Chunk {chunk_idx+1}/{num_chunks}: "
                        f"frames {start_idx}-{end_idx-1}, "
                        f"time range [{chunk_start_time:.3f}, {chunk_end_time:.3f}]")
            
            # 读取该时间范围内的消息
            chunk_data = self._read_chunk_data(bag, chunk_start_time, chunk_end_time)
            
            # 对齐并处理每帧
            for local_idx, (global_idx, main_stamp) in enumerate(
                zip(range(start_idx, end_idx), chunk_timestamps)
            ): 
                aligned_frame = self._align_single_frame(
                    main_stamp=main_stamp,
                    global_idx=global_idx,
                    chunk_data=chunk_data,
                    timestamp_arrays=timestamp_arrays,
                    alignment_indices=alignment_indices,
                    arm_traj_gaps=arm_traj_gaps,
                    last_known_state=last_known_state,   # 传入状态缓存
                    gripper_keys=gripper_keys            # 传入识别的夹爪key集合
                )
                
                frame_callback(aligned_frame, global_idx)
                total_frames += 1
            
            # 释放chunk数据
            del chunk_data
            
            # 调用保存回调
            if save_callback:
                save_callback()
                logger.info(f"Chunk {chunk_idx+1}/{num_chunks} processed and saved. "
                           f"Frames: {start_idx}-{end_idx-1}")
        
        bag.close()
        logger.info(f"Total frames processed: {total_frames}")
        return total_frames
    
    def _precompute_alignment_indices(
        self, 
        main_timestamps: List[float], 
        timestamp_arrays: Dict[str, np.ndarray]
    ) -> Dict[str, List[int]]:
        """
        预计算每个主时间戳对应的各话题索引
        使用二分查找，比每帧都查找快很多
        """
        alignment_indices = {}
        
        for key, ts_array in timestamp_arrays.items():
            if len(ts_array) == 0:
                alignment_indices[key] = []
                continue
            
            indices = []
            for stamp in main_timestamps:
                # 二分查找最近的时间戳
                idx = bisect.bisect_left(ts_array, stamp)
                if idx == 0:
                    closest_idx = 0
                elif idx == len(ts_array):
                    closest_idx = len(ts_array) - 1
                else:
                    # 选择更接近的
                    if abs(ts_array[idx] - stamp) < abs(ts_array[idx-1] - stamp):
                        closest_idx = idx
                    else:
                        closest_idx = idx - 1
                indices.append(closest_idx)
            
            alignment_indices[key] = indices
        
        return alignment_indices
    
    def _detect_arm_traj_gaps(self, all_timestamps: Dict[str, List[float]]) -> List[Tuple[float, float]]:
        """检测kuavo_arm_traj的时间戳间隙"""
        gaps = []
        if not self.only_half_up_body and "action.kuavo_arm_traj" in all_timestamps and len(all_timestamps["action.kuavo_arm_traj"]) > 0:
            timestamps = all_timestamps["action.kuavo_arm_traj"]
            if len(timestamps) > 1:
                gap_threshold = 0.15 * 10 / self.train_hz
                for i in range(1, len(timestamps)):
                    if timestamps[i] - timestamps[i-1] > gap_threshold:
                        gaps.append((timestamps[i-1], timestamps[i]))
        if len(gaps) > 0:
            logger.info(f"Detected {len(gaps)} gaps in action.kuavo_arm_traj")
        return gaps
    
    def _read_chunk_data(self, bag: rosbag.Bag, start_time: float, end_time: float) -> Dict[str, Dict[float, dict]]:
        """
        读取指定时间范围内的消息数据
        
        Returns:
            {topic_key: {timestamp: msg_data}}
        """
        import rospy
        chunk_data = defaultdict(dict)

        topic_to_key = {}
        for k, v in self._topic_process_map.items():
            if v["topic"] not in topic_to_key.keys():
                topic_to_key[v["topic"]] = [k]
            else:
                topic_to_key[v["topic"]].append(k)
        
        # 使用时间范围过滤
        try:
            start_ros_time = rospy.Time.from_sec(start_time)
            end_ros_time = rospy.Time.from_sec(end_time)
            
            for topic, msg, t in bag.read_messages(
                topics=list(topic_to_key.keys()),
                start_time=start_ros_time,
                end_time=end_ros_time
            ):
                keys = topic_to_key.get(topic)
                for key in keys:
                    msg_process_fn = self._topic_process_map[key]["msg_process_fn"]
                    msg_data = msg_process_fn(msg)
                    msg_data["timestamp"] = t.to_sec()
                    chunk_data[key][t.to_sec()] = msg_data
                    # chunk_data[key][msg_data["timestamp"]] = msg_data

        except Exception as e:
            logger.warning(f"Time-range filtering failed: {e}, falling back to full scan")
            # 回退到全量扫描+手动过滤
            for topic, msg, t in bag.read_messages(topics=list(topic_to_key.keys())):
                ts = t.to_sec()
                # ts = msg.header.stamp.to_sec()
                # ts = msg.header.stamp.to_sec() if hasattr(msg, 'header') and hasattr(msg.header, 'stamp') else t.to_sec()

                if start_time <= ts <= end_time:
                    key = topic_to_key.get(topic)
                    if key:
                        msg_process_fn = self._topic_process_map[key]["msg_process_fn"]
                        msg_data = msg_process_fn(msg)
                        msg_data["timestamp"] = ts
                        chunk_data[key][ts] = msg_data
        return dict(chunk_data)
    
    def _align_single_frame(
        self,
        main_stamp: float,
        global_idx: int,
        chunk_data: Dict[str, Dict[float, dict]],
        timestamp_arrays: Dict[str, np.ndarray],
        alignment_indices: Dict[str, List[int]],
        arm_traj_gaps: List[Tuple[float, float]],
        last_known_state: Dict[str, dict],
        gripper_keys: List[str]
    ) -> dict:
        """
        对齐单帧数据
        """
        aligned_frame = {"timestamp": main_stamp}
        
        for key in self._topic_process_map.keys():
            is_gripper = key in gripper_keys
            
            # ================= 1. 特殊处理kuavo_arm_traj的间隙 =================
            if key == "action.kuavo_arm_traj" and arm_traj_gaps:
                in_gap = any(gap_start < main_stamp < gap_end 
                            for gap_start, gap_end in arm_traj_gaps)
                if in_gap:
                    # 在间隙中，使用999填充
                    sample_data = next(iter(chunk_data.get(key, {}).values()), None)
                    if sample_data and "data" in sample_data:
                        data_dim = len(sample_data["data"])
                        aligned_frame[key] = {
                            "data": np.full(data_dim, 999.0, dtype=np.float32),
                            "timestamp": main_stamp
                        }
                    else:
                        aligned_frame[key] = None
                    continue
            
            # 从timestamp_arrays获取对应的时间戳异常处理
            if key not in timestamp_arrays or len(timestamp_arrays[key]) == 0:
                if is_gripper and key in last_known_state:
                    aligned_frame[key] = last_known_state[key].copy()
                    aligned_frame[key]["timestamp"] = main_stamp
                else:
                    aligned_frame[key] = None
                continue

            first_ts = timestamp_arrays[key][0]
            
            # ================= 2. 处理早期尚无数据的情况 =================
            # 逻辑对齐：如果当前主时间戳小于该话题的第一帧时间戳，说明该话题此时还没有数据
            if main_stamp < first_ts:
                # 针对夹爪的特定策略：用全0补全 (在 prefetch 阶段已经设为了全0)
                if is_gripper and key in last_known_state:
                    aligned_frame[key] = last_known_state[key].copy()
                    aligned_frame[key]["timestamp"] = main_stamp
                    continue
                # else:
                #     sample_data = next(iter(chunk_data.get(key, {}).values()), None)
                #     if sample_data and "data" in sample_data and "action." in key and "kuavo_arm_traj" not in key:
                #         logger.warning(f"Topic {key} has no data for timestamp {main_stamp}, filling with zeros. frame idx: {global_idx}")
                #         aligned_frame[key] = {
                #             "data": np.zeros_like(sample_data["data"]),
                #             "timestamp": main_stamp
                #         }
                #     else:
                #         aligned_frame[key] = None
                # continue

            # ================= 3. 数据正常匹配处理 =================
            # 获取预计算的索引
            if key not in alignment_indices or global_idx >= len(alignment_indices[key]):
                if is_gripper and key in last_known_state:
                    aligned_frame[key] = last_known_state[key].copy()
                    aligned_frame[key]["timestamp"] = main_stamp
                else:
                    aligned_frame[key] = None
                continue

            closest_idx = alignment_indices[key][global_idx]
            target_ts = timestamp_arrays[key][closest_idx]
               
            # 从chunk_data中查找数据
            if key in chunk_data:
                # 查找最接近target_ts的数据
                ts_list = list(chunk_data[key].keys())
                if ts_list:
                    closest_chunk_ts = min(ts_list, key=lambda x: abs(x - target_ts))
                    aligned_frame[key] = chunk_data[key][closest_chunk_ts]
                    
                    # 夹爪在获取到了新数据时，更新到 last_known_state 作为后续备份
                    if is_gripper:
                        last_known_state[key] = aligned_frame[key]
                else:
                    aligned_frame[key] = None
            else:
                aligned_frame[key] = None
                
            # ================= 4. 针对夹爪CMD补齐的最终退路 =================
            # 经过上面的逻辑如果没有从 chunk_data 中拿到夹爪数据 (比如夹爪数据在此时段内不发送)
            # 退化利用我们缓存的跨 chunk 的先前状态去替代，保证机械臂在移动时，夹爪不被裁切丢失。
            if is_gripper and aligned_frame[key] is None:
                if key in last_known_state:
                    aligned_frame[key] = last_known_state[key].copy()
                    aligned_frame[key]["timestamp"] = main_stamp
        return aligned_frame
    
    def _load_bag(self, bag_file: str) -> rosbag.Bag:
        """加载rosbag文件"""
        try:
            return rosbag.Bag(bag_file)
        except rosbag.bag.ROSBagUnindexedException:
            logger.warning(f"Bag file {bag_file} is unindexed, attempting to reindex...")
            from .utils import reindex_rosbag
            reindexed_file = reindex_rosbag(bag_file)
            if reindexed_file:
                return rosbag.Bag(reindexed_file)
            else:
                return rosbag.Bag(bag_file, 'r', allow_unindexed=True)
