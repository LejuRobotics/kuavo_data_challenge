import cv2
from torchvision.transforms.functional import to_tensor
from kuavo_data.CvtRosbag2Lerobot import load_raw_episode_data
from client import PolicyClient
import numpy as np
import torch,os,shutil
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.policies.factory import make_pre_post_processors

from pathlib import Path
import kuavo_data.common.kuavo_dataset as kuavo

def read_and_process_episode_data(ep_path):
    def init_param():
        # 手动加载配置
        from omegaconf import OmegaConf
        config_path = Path(__file__).parents[2] / "configs" / "data" / "KuavoRosbag2Lerobot.yaml"
        cfg = OmegaConf.load(config_path)
        
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
        if kuavo.ONLY_HALF_UP_BODY:
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
        else:
            if kuavo.USE_LEJU_CLAW:
                DEFAULT_ARM_JOINT_NAMES = kuavo.DEFAULT_ARM_JOINT_NAMES[:half_arm] + kuavo.DEFAULT_LEJUCLAW_JOINT_NAMES[:half_claw] \
                                        + kuavo.DEFAULT_ARM_JOINT_NAMES[half_arm:] + kuavo.DEFAULT_LEJUCLAW_JOINT_NAMES[half_claw:]
            elif kuavo.USE_QIANGNAO:
                DEFAULT_ARM_JOINT_NAMES = kuavo.DEFAULT_ARM_JOINT_NAMES[:half_arm] + kuavo.DEFAULT_DEXHAND_JOINT_NAMES[:half_dexhand] \
                                        + kuavo.DEFAULT_ARM_JOINT_NAMES[half_arm:] + kuavo.DEFAULT_DEXHAND_JOINT_NAMES[half_dexhand:]             
            DEFAULT_JOINT_NAMES_LIST = kuavo.DEFAULT_LEG_JOINT_NAMES + DEFAULT_ARM_JOINT_NAMES + kuavo.DEFAULT_HEAD_JOINT_NAMES

    init_param()
    imgs_per_cam, state, action, velocity, effort ,claw_state, claw_action,qiangnao_state,qiangnao_action, rq2f85_state, rq2f85_action = load_raw_episode_data(ep_path)
    if len(claw_action)==0 and len(qiangnao_action) == 0:
        claw_action = rq2f85_action
        claw_state = rq2f85_state

    # 对手部进行二值化处理
    if kuavo.IS_BINARY:
        qiangnao_state = np.where(qiangnao_state > 50, 1, 0)
        qiangnao_action = np.where(qiangnao_action > 50, 1, 0)
        claw_state = np.where(claw_state > 50, 1, 0)
        claw_action = np.where(claw_action > 50, 1, 0)
        rq2f85_state = np.where(rq2f85_state > 0.4, 1, 0)
        rq2f85_action = np.where(rq2f85_action > 70, 1, 0)
    else:
        # 进行数据归一化处理
        claw_state = claw_state / 100
        claw_action = claw_action / 100
        qiangnao_state = qiangnao_state / 100
        qiangnao_action = qiangnao_action / 100
        rq2f85_state = rq2f85_state / 0.8
        rq2f85_action = rq2f85_action / 140
    ########################
    # delta 处理
    ########################
    # =====================
    # 为了解决零点问题，将每帧与第一帧相减
    if kuavo.RELATIVE_START:
        # 每个state, action与他们的第一帧相减
        state = state - state[0]
        action = action - action[0]
        
    # ===只处理delta action
    if kuavo.DELTA_ACTION:
        # delta_action = action[1:] - state[:-1]
        # trim = lambda x: x[1:] if (x is not None) and (len(x) > 0) else x
        # state, action, velocity, effort, claw_state, claw_action, qiangnao_state, qiangnao_action = \
        #     map(
        #         trim, 
        #         [state, action, velocity, effort, claw_state, claw_action, qiangnao_state, qiangnao_action]
        #         )
        # for camera, img_array in imgs_per_cam.items():
        #     imgs_per_cam[camera] = img_array[1:]
        # action = delta_action

        # delta_action = np.concatenate(([action[0]-state[0]], action[1:] - action[:-1]), axis=0)
        # action = delta_action

        delta_action = action-state
        action = delta_action
    
    num_frames = state.shape[0]
    frames = []
    for i in range(num_frames):
        if kuavo.ONLY_HALF_UP_BODY:
            if kuavo.USE_LEJU_CLAW:
                # 使用lejuclaw进行上半身关节数据转换
                if kuavo.CONTROL_HAND_SIDE == "left" or kuavo.CONTROL_HAND_SIDE == "both":
                    output_state = state[i, kuavo.SLICE_ROBOT[0][0]:kuavo.SLICE_ROBOT[0][-1]]
                    output_state = np.concatenate((output_state, claw_state[i, kuavo.SLICE_CLAW[0][0]:kuavo.SLICE_CLAW[0][-1]].astype(np.float32)), axis=0)
                    output_action = action[i, kuavo.SLICE_ROBOT[0][0]:kuavo.SLICE_ROBOT[0][-1]]
                    output_action = np.concatenate((output_action, claw_action[i, kuavo.SLICE_CLAW[0][0]:kuavo.SLICE_CLAW[0][-1]].astype(np.float32)), axis=0)
                if kuavo.CONTROL_HAND_SIDE == "right" or kuavo.CONTROL_HAND_SIDE == "both":
                    if kuavo.CONTROL_HAND_SIDE == "both":
                        output_state = np.concatenate((output_state, state[i, kuavo.SLICE_ROBOT[1][0]:kuavo.SLICE_ROBOT[1][-1]]), axis=0)
                        output_state = np.concatenate((output_state, claw_state[i, kuavo.SLICE_CLAW[1][0]:kuavo.SLICE_CLAW[1][-1]].astype(np.float32)), axis=0)
                        output_action = np.concatenate((output_action, action[i, kuavo.SLICE_ROBOT[1][0]:kuavo.SLICE_ROBOT[1][-1]]), axis=0)
                        output_action = np.concatenate((output_action, claw_action[i, kuavo.SLICE_CLAW[1][0]:kuavo.SLICE_CLAW[1][-1]].astype(np.float32)), axis=0)
                    else:
                        output_state = state[i, kuavo.SLICE_ROBOT[1][0]:kuavo.SLICE_ROBOT[1][-1]]
                        output_state = np.concatenate((output_state, claw_state[i, kuavo.SLICE_CLAW[1][0]:kuavo.SLICE_CLAW[1][-1]].astype(np.float32)), axis=0)
                        output_action = action[i, kuavo.SLICE_ROBOT[1][0]:kuavo.SLICE_ROBOT[1][-1]]
                        output_action = np.concatenate((output_action, claw_action[i, kuavo.SLICE_CLAW[1][0]:kuavo.SLICE_CLAW[1][-1]].astype(np.float32)), axis=0)

            elif kuavo.USE_QIANGNAO:
                # 类型: kuavo_sdk/robotHandPosition
                # left_hand_position (list of float): 左手位置，包含6个元素，每个元素的取值范围为[0, 100], 0 为张开，100 为闭合。
                # right_hand_position (list of float): 右手位置，包含6个元素，每个元素的取值范围为[0, 100], 0 为张开，100 为闭合。
                # 构造qiangnao类型的output_state的数据结构的长度应该为26
                if kuavo.CONTROL_HAND_SIDE == "left" or kuavo.CONTROL_HAND_SIDE == "both":
                    output_state = state[i, kuavo.SLICE_ROBOT[0][0]:kuavo.SLICE_ROBOT[0][-1]]
                    output_state = np.concatenate((output_state, qiangnao_state[i, kuavo.SLICE_DEX[0][0]:kuavo.SLICE_DEX[0][-1]].astype(np.float32)), axis=0)

                    output_action = action[i, kuavo.SLICE_ROBOT[0][0]:kuavo.SLICE_ROBOT[0][-1]]
                    output_action = np.concatenate((output_action, qiangnao_action[i, kuavo.SLICE_DEX[0][0]:kuavo.SLICE_DEX[0][-1]].astype(np.float32)), axis=0)
                if kuavo.CONTROL_HAND_SIDE == "right" or kuavo.CONTROL_HAND_SIDE == "both":
                    if kuavo.CONTROL_HAND_SIDE == "both":
                        output_state = np.concatenate((output_state, state[i, kuavo.SLICE_ROBOT[1][0]:kuavo.SLICE_ROBOT[1][-1]]), axis=0)
                        output_state = np.concatenate((output_state, qiangnao_state[i, kuavo.SLICE_DEX[1][0]:kuavo.SLICE_DEX[1][-1]].astype(np.float32)), axis=0)
                        output_action = np.concatenate((output_action, action[i, kuavo.SLICE_ROBOT[1][0]:kuavo.SLICE_ROBOT[1][-1]]), axis=0)
                        output_action = np.concatenate((output_action, qiangnao_action[i, kuavo.SLICE_DEX[1][0]:kuavo.SLICE_DEX[1][-1]].astype(np.float32)), axis=0)
                    else:
                        output_state = state[i, kuavo.SLICE_ROBOT[1][0]:kuavo.SLICE_ROBOT[1][-1]]
                        output_state = np.concatenate((output_state, qiangnao_state[i, kuavo.SLICE_DEX[1][0]:kuavo.SLICE_DEX[1][-1]].astype(np.float32)), axis=0)
                        output_action = action[i, kuavo.SLICE_ROBOT[1][0]:kuavo.SLICE_ROBOT[1][-1]]
                        output_action = np.concatenate((output_action, qiangnao_action[i, kuavo.SLICE_DEX[1][0]:kuavo.SLICE_DEX[1][-1]].astype(np.float32)), axis=0)
                # output_action = np.concatenate((output_action, action[i, 26:28]), axis=0)
        else:
            if kuavo.USE_LEJU_CLAW:
                # 使用lejuclaw进行全身关节数据转换
                # 原始的数据是28个关节的数据对应原始的state和action数据的长度为28
                # 数据顺序:
                # 前 12 个数据为下肢电机数据:
                #     0~5 为左下肢数据 (l_leg_roll, l_leg_yaw, l_leg_pitch, l_knee, l_foot_pitch, l_foot_roll)
                #     6~11 为右下肢数据 (r_leg_roll, r_leg_yaw, r_leg_pitch, r_knee, r_foot_pitch, r_foot_roll)
                # 接着 14 个数据为手臂电机数据:
                #     12~18 左臂电机数据 ("l_arm_pitch", "l_arm_roll", "l_arm_yaw", "l_forearm_pitch", "l_hand_yaw", "l_hand_pitch", "l_hand_roll")
                #     19~25 为右臂电机数据 ("r_arm_pitch", "r_arm_roll", "r_arm_yaw", "r_forearm_pitch", "r_hand_yaw", "r_hand_pitch", "r_hand_roll")
                # 最后 2 个为头部电机数据: head_yaw 和 head_pitch
                
                # TODO：构造目标切片
                output_state = state[i, 0:19]
                output_state = np.insert(output_state, 19, claw_state[i, 0].astype(np.float32))
                output_state = np.concatenate((output_state, state[i, 19:26]), axis=0)
                output_state = np.insert(output_state, 19, claw_state[i, 1].astype(np.float32))
                output_state = np.concatenate((output_state, state[i, 26:28]), axis=0)

                output_action = action[i, 0:19]
                output_action = np.insert(output_action, 19, claw_action[i, 0].astype(np.float32))
                output_action = np.concatenate((output_action, action[i, 19:26]), axis=0)
                output_action = np.insert(output_action, 19, claw_action[i, 1].astype(np.float32))
                output_action = np.concatenate((output_action, action[i, 26:28]), axis=0)

            elif kuavo.USE_QIANGNAO:
                output_state = state[i, 0:19]
                output_state = np.concatenate((output_state, qiangnao_state[i, 0:6].astype(np.float32)), axis=0)
                output_state = np.concatenate((output_state, state[i, 19:26]), axis=0)
                output_state = np.concatenate((output_state, qiangnao_state[i, 6:12].astype(np.float32)), axis=0)
                output_state = np.concatenate((output_state, state[i, 26:28]), axis=0)

                output_action = action[i, 0:19]
                output_action = np.concatenate((output_action, qiangnao_action[i, 0:6].astype(np.float32)),axis=0)
                output_action = np.concatenate((output_action, action[i, 19:26]), axis=0)
                output_action = np.concatenate((output_action, qiangnao_action[i, 6:12].astype(np.float32)), axis=0)
                output_action = np.concatenate((output_action, action[i, 26:28]), axis=0)  
        frame = {
            "observation.state": torch.from_numpy(output_state).type(torch.float32),
            "action": torch.from_numpy(output_action).type(torch.float32),
        }
        
        for camera, img_array in imgs_per_cam.items():
            if "depth" in camera:
                # frame[f"observation.{camera}"] = img_array[i]
                min_depth, max_dpeth = kuavo.DEPTH_RANGE[0], kuavo.DEPTH_RANGE[1]
                frame[f"observation.{camera}"] = np.clip(img_array[i], min_depth, max_dpeth)
                print("[info]: Clip depth in range %d ~ %d"%(min_depth, max_dpeth))
            else:
                frame[f"observation.images.{camera}"] = img_array[i]
        
        if velocity is not None:
            frame["observation.velocity"] = velocity[i]
        if effort is not None:
            frame["observation.effort"] = effort[i]
        frames.append(frame)
    
    return frames

def img_preprocess(image, device="cpu"):
    return to_tensor(image).unsqueeze(0).to(device, non_blocking=True)

def depth_preprocess(depth, device="cpu",depth_range=[0,1000]):
    depth_uint16 =  torch.tensor(depth,dtype=torch.float32).clamp(*depth_range).unsqueeze(0).unsqueeze(0).to(device, non_blocking=True)
    max_depth = depth_uint16.max()
    min_depth = depth_uint16.min()
    depth_normalized = (depth_uint16 - min_depth) / (max_depth - min_depth + 1e-9)  # 归一化到 [0, 1]
    # depth_normalized = (depth_normalized * 255).astype(np.uint8)
    return depth_normalized
    
# Convert raw observations into the observations required by the model
def hardware_obses_to_policy_obs_dict(obs):
    device = 'cuda'
    obs_dict = {}
    for k,v in obs.items():
        if "images" in k:
            obs_dict[k] = img_preprocess(v, device=device)
        elif "state" in k:
            obs_dict[k] = torch.tensor(v,dtype=torch.float32).unsqueeze(0).to(device, non_blocking=True)
            print(obs_dict[k].shape)
        elif "depth" in k:
            obs_dict[k] = depth_preprocess(v, device=device, depth_range=[0,1500])
    return obs_dict

def main(ep_path="/home/ubun-new/go_bag/bag_for_handover/A10-A01-206-208-92-71-dex_hand-20250930100830-v1.bag"):
    policy_client = PolicyClient()
    frames = read_and_process_episode_data(ep_path) # 加载一条遥操数据，每步为一个字典 包含observation.state, observation.images.head_cam_h, observation.images.wrist_cam_l, observation.images.wrist_cam_r, action
    preprocessor, postprocessor = make_pre_post_processors(None,"outputs/train/test_handover/state_fuse/run_1008/epochbest")
    for i in range(len(frames)):
    	# 将真机的观测处理成字典
        obs_dict = hardware_obses_to_policy_obs_dict(frames[i])
        # print("head img",obs_dict["observation.depth_h"].max())

        obs_dict = preprocessor(obs_dict)

        print("head img",obs_dict.keys())
        print("head img",obs_dict["observation.images.head_cam_h"].min(), obs_dict["observation.images.head_cam_h"].max())
        print("head img",obs_dict["observation.depth_h"].min(), obs_dict["observation.depth_h"].max())
        # raise ValueError()

        action_pred = policy_client.select_action(obs_dict) # numpy(26,)维动作，按照kuavo转lerobot的方式 由7维左臂+6维左手+7维右臂+6维右手组成。其中手臂的动作为目标关节角，灵巧手的动作在[0,1]之间、需要乘以100再发给真机
        action_pred = postprocessor(action_pred).squeeze(0).cpu().numpy()
        action_groundtruth = frames[i]["action"].cpu().numpy()
        print("Timestep:", i, "Action MSE:", ((action_pred - action_groundtruth)**2).mean())
        # print("action_pred:",action_pred)
        # print("action_groundtruth:",action_groundtruth)

if __name__ == "__main__":
    main()
