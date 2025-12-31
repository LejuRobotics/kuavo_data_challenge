"""
Script to convert Kuavo rosbag data to the LeRobot dataset v2.0 format.

Example usage: uv run examples/aloha_real/convert_aloha_data_to_lerobot.py --raw-dir /path/to/raw/data --repo-id <org>/<dataset-name>
"""
import lerobot_patches.custom_patches  # Ensure custom patches are applied, DON'T REMOVE THIS LINE!
import dataclasses
from pathlib import Path
import shutil
import hydra
from omegaconf import DictConfig
from typing import Literal
import sys
import os
from rich.logging import RichHandler
import logging
import resource

logging.basicConfig(
    level="DEBUG",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)


from pympler import asizeof
import matplotlib.pyplot as plt



# 临时测试使用：
# -------------------------------------------------
DEFAULT_ARM_JOINT_RANGE = [
    [-3.14159, 1.5708], 
    [-0.349066, 2.0944], 
    [-1.5708, 1.5708],
    [-2.61799, 0],
    [-1.5708, 1.5708], 
    [-1.309, 0.698132], 
    [-0.698132, 0.698132],
    [-1, 1],
    [-3.14159, 1.5708], 
    [-2.0944, 0.349066], 
    [-1.5708, 1.5708], 
    [-2.61799, 0], 
    [-1.5708, 1.5708], 
    [-0.698132, 1.309], 
    [-0.698132, 0.698132],
    [-1, 1]
] 
# -------------------------------------------------

def get_attr_sizes(obj, prefix=""):
    """递归获取对象每个属性及嵌套属性的内存占用"""
    sizes = {}
    for attr in dir(obj):
        if attr.startswith("__"):
            continue
        try:
            value = getattr(obj, attr)
        except Exception:
            continue
        key = f"{prefix}.{attr}" if prefix else attr
        size = asizeof.asizeof(value)
        sizes[key] = size
        # 如果是自定义类实例，递归获取
        if hasattr(value, "__dict__"):
            sizes.update(get_attr_sizes(value, prefix=key))
    return sizes

def visualize_memory(attr_sizes, top_n=20):
    """可视化内存占用"""
    # 按大小排序，取前 top_n
    sorted_attrs = sorted(attr_sizes.items(), key=lambda x: x[1], reverse=True)[:top_n]
    labels, sizes = zip(*sorted_attrs)
    sizes_kb = [s / 1024 /1024 for s in sizes]

    plt.figure(figsize=(12, 6))
    plt.barh(labels[::-1], sizes_kb[::-1])
    plt.xlabel("Memory (MB)")
    plt.title(f"Top {top_n} attributes by memory usage")
    plt.tight_layout()
    plt.show()




log_print = logging.getLogger("rich")

try:
    from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    log_print.warning("import lerobot.common.xxx will be deprecated in lerobot v2.0, please use lerobot.xxx instead in the future.")
except Exception as import_error:
    try:
        import lerobot
    except Exception as lerobot_error:
        log_print.error("Error: lerobot package not found. Please change to 'third_party/lerobot' and install it using 'pip install -e .'.")
        sys.exit(1)
    log_print.info("Error: "+ str(import_error))
    log_print.info("Import lerobot.common.xxx is deprecated in lerobot v2.0, try to use import lerobot.xxx instead ...")
    try:
        from lerobot.datasets.lerobot_dataset import LEROBOT_HOME
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        log_print.info("import lerobot.datasets.lerobot_dataset ok!")
    except Exception as import_failed:
        log_print.info("Error:"+str(import_failed))
        if "LEROBOT_HOME" in str(import_failed):
            try:
                from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME as LEROBOT_HOME
                from lerobot.datasets.lerobot_dataset import LeRobotDataset
                log_print.info("import lerobot.datasets.lerobot_dataset HF_LEROBOT_HOME,  LeRobotDataset ok!")
            except Exception as e:
                log_print.error(str(e))
                sys.exit(1)


import numpy as np
import torch
import tqdm
import json

import kuavo_data.common.kuavo_dataset as kuavo
import rospy

@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None

DEFAULT_DATASET_CONFIG = DatasetConfig()


def get_cameras(bag_data: dict) -> list[str]:
    """
    /cam_l/color/image_raw/compressed                    : sensor_msgs/CompressedImage                
    /cam_r/color/image_raw/compressed                    : sensor_msgs/CompressedImage                
    /zedm/zed_node/left/image_rect_color/compressed      : sensor_msgs/CompressedImage                
    /zedm/zed_node/right/image_rect_color/compressed     : sensor_msgs/CompressedImage 
    """
    cameras = []

    for k in kuavo.DEFAULT_CAMERA_NAMES:
        cameras.append(k)
    return cameras

def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    root: str,
) -> LeRobotDataset:
    
    # 根据config的参数决定是否为半身和末端的关节类型
    motors = DEFAULT_JOINT_NAMES_LIST
    # TODO: auto detect cameras
    cameras = kuavo.DEFAULT_CAMERA_NAMES


    action_dim = (len(motors),)

    # set action name/dim, state name/dim,
    action_name =  motors

    state_dim = (len(motors),)


    state_name = kuavo.DEFAULT_ARM_JOINT_NAMES[:len(kuavo.DEFAULT_ARM_JOINT_NAMES)//2] + ["gripper_l"] + kuavo.DEFAULT_ARM_JOINT_NAMES[len(kuavo.DEFAULT_ARM_JOINT_NAMES)//2:] + ["gripper_r"]
    
    if not kuavo.ONLY_HALF_UP_BODY:
        action_dim = (action_dim[0] + 3 + 1,)  # cmd_pos_world3+断点标志1
        action_name += ["cmd_pos_x", "cmd_pos_y", "cmd_pos_yaw", "ctrl_change_cmd"]
        state_dim = (state_dim[0] + 0,)  # 机器人base_pos_world3+断点标志1
        state_name += []  # 如上 ["base_pos_x", "base_pos_y", "base_pos_yaw", "ctrl_change_flag"]

    # create corresponding features
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": state_dim,
            "names": {
                "state_names": state_name
            }
        },
        "action": {
            "dtype": "float32",
            "shape": action_dim,
            "names": {
                "action_names": action_name
            }
        },
    }

    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    for cam in cameras:
        if 'depth' in cam:
            features[f"observation.{cam}"] = {
                "dtype": mode, 
                "shape": (3, kuavo.RESIZE_H, kuavo.RESIZE_W),  # Attention: for datasets.features "image" and "video", it must be c,h,w style! 
                "names": [
                    "channels",
                    "height",
                    "width",
                ],
            }
        else:
            features[f"observation.images.{cam}"] = {
                "dtype": mode,
                "shape": (3, kuavo.RESIZE_H, kuavo.RESIZE_W),
                "names": [
                    "channels",
                    "height",
                    "width",
                ],
            }

    if Path(LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=kuavo.TRAIN_HZ,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
        root=root,
    )

def load_raw_images_per_camera(bag_data: dict) -> dict[str, np.ndarray]:
    imgs_per_cam = {}
    for camera in get_cameras(bag_data):
        imgs_per_cam[camera] = np.array([msg['data'] for msg in bag_data[camera]])
        print(f"camera {camera} image", imgs_per_cam[camera].shape)
    
    return imgs_per_cam


def load_raw_episode_data(
    ep_path: Path,
) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:

    bag_reader = kuavo.KuavoRosbagReader()
    bag_data = bag_reader.process_rosbag(ep_path)
    
    state = np.array([msg['data'] for msg in bag_data['observation.state']], dtype=np.float32)
    action = np.array([msg['data'] for msg in bag_data['action']], dtype=np.float32)
    action_kuavo_arm_traj = np.array([msg['data'] for msg in bag_data['action.kuavo_arm_traj']], dtype=np.float32)
    claw_state = np.array([msg['data'] for msg in bag_data['observation.claw']], dtype=np.float64)
    claw_action= np.array([msg['data'] for msg in bag_data['action.claw']], dtype=np.float64)
    qiangnao_state = np.array([msg['data'] for msg in bag_data['observation.qiangnao']], dtype=np.float64)
    qiangnao_action= np.array([msg['data'] for msg in bag_data['action.qiangnao']], dtype=np.float64)
    rq2f85_state = np.array([msg['data'] for msg in bag_data['observation.rq2f85']], dtype=np.float64)
    rq2f85_action= np.array([msg['data'] for msg in bag_data['action.rq2f85']], dtype=np.float64)
    cmd_pos_world_action = np.array([msg['data'] for msg in bag_data['action.cmd_pos_world']], dtype=np.float32)
    action_kuavo_arm_traj_alt = np.array([msg['data'] for msg in bag_data['action.kuavo_arm_traj_alt']], dtype=np.float32)
    # print("eef_type shape: ",claw_action.shape,qiangnao_action.shape, rq2f85_action.shape)
    action[:, 12:26] = action_kuavo_arm_traj if len(action_kuavo_arm_traj_alt) == 0 else action_kuavo_arm_traj_alt

    velocity = None
    effort = None
    
    imgs_per_cam = load_raw_images_per_camera(bag_data)
    
    return imgs_per_cam, state, action, velocity, effort ,claw_state ,claw_action,qiangnao_state,qiangnao_action, rq2f85_state, rq2f85_action, cmd_pos_world_action, action_kuavo_arm_traj,


def diagnose_frame_data(data):
    for k, v in data.items():
        print(f"Field: {k}")
        print(f"  Shape    : {v.shape}")
        print(f"  Dtype    : {v.dtype}")
        print(f"  Type     : {type(v).__name__}")
        print("-" * 40)


def populate_dataset(
    dataset: LeRobotDataset,
    bag_files: list[Path],
    task: str,
    episodes: list[int] | None = None,
) -> LeRobotDataset:
    if episodes is None:
        episodes = range(len(bag_files))
    failed_bags = []
    print( f"Total episodes to process: {len(episodes)}")

    for ep_idx in tqdm.tqdm(episodes):
        ep_path = bag_files[ep_idx]
        from termcolor import colored
        print(colored(f"Processing {ep_path}", "yellow", attrs=["bold"]))
        # 默认读取所有的数据如果话题不存在相应的数值应该是一个空的数据
        # try:
        imgs_per_cam, state, action, velocity, effort ,claw_state, claw_action,qiangnao_state,qiangnao_action, rq2f85_state, rq2f85_action, cmd_pos_world_action, action_kuavo_arm_traj = load_raw_episode_data(ep_path)
        # except Exception as e:
        #     print(f"❌ Error processing {ep_path}: {e}")
        #     failed_bags.append(str(ep_path))
        #     continue
        # 对手部进行二值化处理
        if kuavo.IS_BINARY:
            qiangnao_state = np.where(qiangnao_state > 50, 1, 0)
            qiangnao_action = np.where(qiangnao_action > 50, 1, 0)
            claw_state = np.where(claw_state > 50, 1, 0)
            claw_action = np.where(claw_action > 50, 1, 0)
            rq2f85_state = np.where(rq2f85_state > 0.4, 1, 0)
            rq2f85_action = np.where(rq2f85_action > 70, 1, 0)

            # rq2f85_state = np.where(rq2f85_state > 0.1, 1, 0)
            # rq2f85_action = np.where(rq2f85_action > 128, 1, 0)
        else:
            # 进行数据归一化处理
            claw_state = claw_state / 100
            claw_action = claw_action / 100
            qiangnao_state = qiangnao_state / 100
            qiangnao_action = qiangnao_action / 100
            rq2f85_state = rq2f85_state / 0.8
            rq2f85_action = rq2f85_action / 255

            # rq2f85_state = rq2f85_state / 0.8
            # rq2f85_action = rq2f85_action / 255
        print(f"eef_action shape, leju_claw: {claw_action.shape},qiangnao: {qiangnao_action.shape}, rq2f85: {rq2f85_action.shape}")
        if len(claw_action)==0 and len(qiangnao_action) == 0:
            claw_action = rq2f85_action
            claw_state = rq2f85_state

        # =====================
        # 为了解决零点问题，将每帧与第一帧相减
        if kuavo.RELATIVE_START:
            # 每个state, action与他们的第一帧相减
            state = state - state[0]
            action = action - action[0]
        

        def get_hand_data(i, hand_side, hand_type):
            if hand_type == "LEJU":
                s_slice = kuavo.SLICE_ROBOT[hand_side]
                c_slice = kuavo.SLICE_CLAW[hand_side]
                s = np.concatenate((state[i, s_slice[0]:s_slice[-1]], claw_state[i, c_slice[0]:c_slice[-1]]))
                a = np.concatenate((action[i, s_slice[0]:s_slice[-1]], claw_action[i, c_slice[0]:c_slice[-1]]))
            else:
                s_slice = kuavo.SLICE_ROBOT[hand_side]
                d_slice = kuavo.SLICE_DEX[hand_side]
                s = np.concatenate((state[i, s_slice[0]:s_slice[-1]], qiangnao_state[i, d_slice[0]:d_slice[-1]]))
                a = np.concatenate((action[i, s_slice[0]:s_slice[-1]], qiangnao_action[i, d_slice[0]:d_slice[-1]]))
            return s, a

        num_frames = state.shape[0]

        for i in range(num_frames):
            if kuavo.USE_LEJU_CLAW or kuavo.USE_QIANGNAO:
                hand_type = "LEJU" if kuavo.USE_LEJU_CLAW else "QIANGNAO"
                s_list, a_list = [], []
                if kuavo.CONTROL_HAND_SIDE in ("left", "both"):
                    s, a = get_hand_data(i, 0, hand_type)
                    s_list.append(s); a_list.append(a)
                if kuavo.CONTROL_HAND_SIDE in ("right", "both"):
                    s, a = get_hand_data(i, 1, hand_type)
                    s_list.append(s); a_list.append(a)
                output_state = np.concatenate(s_list).astype(np.float32)
                output_action = np.concatenate(a_list).astype(np.float32)

            # ~~~~~~~~~~~~~~~~~~~~~~~~~手臂关节角度范围限制，防止有数据超限 ~~~~~~~~~~~~~~~~~~~~~~~~~
            assert len(DEFAULT_ARM_JOINT_RANGE) >= 16, "DEFAULT_ARM_JOINT_RANGE should have at least 16 joint ranges"

            if kuavo.CONTROL_HAND_SIDE == "left":
                joint_indices = range(0, 8)
            elif kuavo.CONTROL_HAND_SIDE == "right":
                joint_indices = range(8, 16)
            elif kuavo.CONTROL_HAND_SIDE == "both":
                joint_indices = range(0, 16)
            else:
                raise ValueError(f"Invalid CONTROL_HAND_SIDE: {kuavo.CONTROL_HAND_SIDE}")

            # 保证 output_action 长度匹配选中的手臂
            assert len(joint_indices) == output_action.shape[0], (
                f"Expected output_action of length {len(joint_indices)}, "
                f"but got {output_action.shape[0]}"
            )

            for enu_i, jidx_k in enumerate(joint_indices):
                low, high = DEFAULT_ARM_JOINT_RANGE[jidx_k]
                if output_action[enu_i] < low:
                    output_action[enu_i] = low
                elif output_action[enu_i] > high:
                    output_action[enu_i] = high
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                
            final_action = output_action
            final_state = output_state


            # ~~~~~~~~~~~~~~~~~~~~~~~~~deal with cmd_pos_world and gap_flag under task4 ~~~~~~~~~~~~~~~~~~~~~~~~~
            if not kuavo.ONLY_HALF_UP_BODY:
                cmd_pos_world = cmd_pos_world_action[i]
                # 6. 断点标志 (1维): 检查action_kuavo_arm_traj是否包含999
                gap_flag = 1.0 if np.any(action_kuavo_arm_traj[i] == 999.0) else 0.0
                # 合并所有action
                final_action = np.concatenate([
                    final_action,
                    cmd_pos_world,
                    np.array([gap_flag], dtype=np.float32)
                ], axis=0)
            
            frame = {
                "observation.state": torch.from_numpy(final_state).type(torch.float32),      # left+right: pos+rot6d+gripper;  dim: 2*10           
                "action": torch.from_numpy(final_action).type(torch.float32),                # left+right: pos+rot6d+gripper;  dim: 2*10                  
            }

            for idx, (camera, img_array) in enumerate(imgs_per_cam.items()):
                if "depth" in camera:
                    min_depth, max_depth = kuavo.DEPTH_RANGE[0], kuavo.DEPTH_RANGE[1]
                    depth_uint16 = np.clip(img_array[i], min_depth, max_depth)
                    max_depth = depth_uint16.max()
                    min_depth = depth_uint16.min()
                    depth_normalized = (depth_uint16 - min_depth) / (max_depth - min_depth + 1e-9)
                    depth_normalized = (depth_normalized * 255).astype(np.uint8)
                    frame[f"observation.{camera}"] = depth_normalized[..., np.newaxis].repeat(3, axis=-1)
                    if i % 50 == 0:
                        print("[info]: Clip depth in range %d ~ %d, camera: %s" % (min_depth, max_depth, camera))

                else:
                    frame[f"observation.images.{camera}"] = img_array[i]
            
            if velocity is not None:
                frame["observation.velocity"] = velocity[i]
            if effort is not None:
                frame["observation.effort"] = effort[i]

            frame["task"] = task
            # diagnose_frame_data(frame)
            dataset.add_frame(frame)
        # dataset.save_episode(task="Pick the black workpiece from the white conveyor belt on your left and place it onto the white box in front of you",)
        # raise ValueError("stop!")


        # usage = resource.getrusage(resource.RUSAGE_SELF)
        # print(f"~~~~~~~~~~~~~~Before Memory usage: {usage.ru_maxrss / 1024:.2f} MB")
        # print(dataset.episode_buffer)
        dataset.save_episode()
        # usage = resource.getrusage(resource.RUSAGE_SELF)
        # print(f"~~~~~~~~~~~~~~After Memory usage: {usage.ru_maxrss / 1024:.2f} MB")
        # print(dataset.episode_buffer)
        # sizes = get_attr_sizes(dataset)
        # for k, v in sizes.items():
        #     print(f"{k}: {v/1024/1024:.2f} MB")
        # visualize_memory(sizes, top_n=10)
        # dataset.hf_dataset = None  # reduce memory usage in data convert
        # del dataset.hf_dataset
        dataset.hf_dataset = dataset.create_hf_dataset()  # reset to reduce memory usage in data convert

    # 将失败的bag文件写入error.txt
    if failed_bags:
        with open("error.txt", "w") as f:
            for bag in failed_bags:
                f.write(bag + "\n")
        print(f"❌ {len(failed_bags)} failed bags written to error.txt")

    return dataset
            


def port_kuavo_rosbag(
    raw_dir: Path,
    repo_id: str,
    raw_repo_id: str | None = None,
    task: str = "DEBUG",
    *,
    episodes: list[int] | None = None,
    push_to_hub: bool = False,
    is_mobile: bool = False,
    mode: Literal["video", "image"] = "video",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    root: str,
    n: int | None = None,
):
    # Download raw data if not exists
    if (LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    bag_reader = kuavo.KuavoRosbagReader()
    bag_files = bag_reader.list_bag_files(raw_dir)
    
    if isinstance(n, int) and n > 0:
        num_available_bags = len(bag_files)
        if n > num_available_bags:
            log_print.warning(f"Warning: Requested {n} bags, but only {num_available_bags} are available. Using all available bags.")
            n = num_available_bags
        
        # random sample num_of_bag files
        select_idx = np.random.choice(num_available_bags, n, replace=False)
        bag_files = [bag_files[i] for i in select_idx]
    
    dataset = create_empty_dataset( 
        repo_id,
        robot_type="kuavo4pro",
        mode=mode,
        has_effort=False,
        has_velocity=False,
        dataset_config=dataset_config,
        root = root,
    )
    dataset = populate_dataset(
        dataset,
        bag_files,
        task=task,
        episodes=episodes,
    )
    # dataset.consolidate()
    
@hydra.main(config_path="../configs/data/", config_name="KuavoRosbag2Lerobot", version_base=None)
def main(cfg: DictConfig):

    global DEFAULT_JOINT_NAMES_LIST
    kuavo.init_parameters(cfg)

    n = cfg.rosbag.num_used
    raw_dir = cfg.rosbag.rosbag_dir
    version = cfg.rosbag.lerobot_dir

    task_name = os.path.basename(raw_dir)
    repo_id = f'lerobot/{task_name}'
    lerobot_dir = os.path.join(raw_dir,"../",version,"lerobot")
    if os.path.exists(lerobot_dir):
        shutil.rmtree(lerobot_dir)
    
    half_arm = len(kuavo.DEFAULT_ARM_JOINT_NAMES) // 2
    half_claw = len(kuavo.DEFAULT_LEJUCLAW_JOINT_NAMES) // 2
    half_dexhand = len(kuavo.DEFAULT_DEXHAND_JOINT_NAMES) // 2
    UP_START_INDEX = 12
    # if kuavo.ONLY_HALF_UP_BODY:
    if kuavo.USE_LEJU_CLAW:
        DEFAULT_ARM_JOINT_NAMES = kuavo.DEFAULT_ARM_JOINT_NAMES[:half_arm] + kuavo.DEFAULT_LEJUCLAW_JOINT_NAMES[:half_claw] \
                                + kuavo.DEFAULT_ARM_JOINT_NAMES[half_arm:] + kuavo.DEFAULT_LEJUCLAW_JOINT_NAMES[half_claw:]
        arm_slice = [
            (kuavo.SLICE_ROBOT[0][0] - UP_START_INDEX, kuavo.SLICE_ROBOT[0][-1] - UP_START_INDEX),(kuavo.SLICE_CLAW[0][0] + half_arm, kuavo.SLICE_CLAW[0][-1] + half_arm), 
            (kuavo.SLICE_ROBOT[1][0] - UP_START_INDEX + half_claw, kuavo.SLICE_ROBOT[1][-1] - UP_START_INDEX + half_claw), (kuavo.SLICE_CLAW[1][0] + half_arm * 2, kuavo.SLICE_CLAW[1][-1] + half_arm * 2)
            ]
    elif kuavo.USE_QIANGNAO:  
        DEFAULT_ARM_JOINT_NAMES = kuavo.DEFAULT_ARM_JOINT_NAMES[:half_arm] + kuavo.DEFAULT_DEXHAND_JOINT_NAMES[:half_dexhand] \
                                + kuavo.DEFAULT_ARM_JOINT_NAMES[half_arm:] + kuavo.DEFAULT_DEXHAND_JOINT_NAMES[half_dexhand:]               
        arm_slice = [
            (kuavo.SLICE_ROBOT[0][0] - UP_START_INDEX, kuavo.SLICE_ROBOT[0][-1] - UP_START_INDEX),(kuavo.SLICE_DEX[0][0] + half_arm, kuavo.SLICE_DEX[0][-1] + half_arm), 
            (kuavo.SLICE_ROBOT[1][0] - UP_START_INDEX + half_dexhand, kuavo.SLICE_ROBOT[1][-1] - UP_START_INDEX + half_dexhand), (kuavo.SLICE_DEX[1][0] + half_arm * 2, kuavo.SLICE_DEX[1][-1] + half_arm * 2)
            ]
    DEFAULT_JOINT_NAMES_LIST = [DEFAULT_ARM_JOINT_NAMES[k] for l, r in arm_slice for k in range(l, r)]  
    # else:
    #     if kuavo.USE_LEJU_CLAW:
    #         DEFAULT_ARM_JOINT_NAMES = kuavo.DEFAULT_ARM_JOINT_NAMES[:half_arm] + kuavo.DEFAULT_LEJUCLAW_JOINT_NAMES[:half_claw] \
    #                                 + kuavo.DEFAULT_ARM_JOINT_NAMES[half_arm:] + kuavo.DEFAULT_LEJUCLAW_JOINT_NAMES[half_claw:]
    #     elif kuavo.USE_QIANGNAO:
    #         DEFAULT_ARM_JOINT_NAMES = kuavo.DEFAULT_ARM_JOINT_NAMES[:half_arm] + kuavo.DEFAULT_DEXHAND_JOINT_NAMES[:half_dexhand] \
    #                                 + kuavo.DEFAULT_ARM_JOINT_NAMES[half_arm:] + kuavo.DEFAULT_DEXHAND_JOINT_NAMES[half_dexhand:]             
    #     DEFAULT_JOINT_NAMES_LIST = kuavo.DEFAULT_LEG_JOINT_NAMES + DEFAULT_ARM_JOINT_NAMES + kuavo.DEFAULT_HEAD_JOINT_NAMES

    port_kuavo_rosbag(raw_dir, repo_id, root=lerobot_dir,n = n, task=kuavo.TASK_DESCRIPTION)



if __name__ == "__main__":
    
    main()
    

    