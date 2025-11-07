#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import rospy
from kuavo_humanoid_sdk.msg.kuavo_msgs.srv import fkSrv

from kuavo_data.common.R_Transform import Transform
# from transform_util import Transform

def fk_srv_client(joint_angles: list):
    try:
        fk_srv = rospy.ServiceProxy('/ik/fk_srv', fkSrv)
        fk_result = fk_srv(joint_angles)
        # print("FK result:", fk_result.success)
        return fk_result.hand_poses
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)
        return None

def joint_angle_2_eep6d(joint_angles: list)-> np.ndarray:
    hand_poses = fk_srv_client(joint_angles)

    if hand_poses is None:
        hand_poses = fk_srv_client(joint_angles)
        if hand_poses is None:
            raise ValueError("FK service call failed")
    
    left_pos = hand_poses.left_pose.pos_xyz            # type: tuple
    left_quant_xyzw = hand_poses.left_pose.quat_xyzw   # type: tuple
    # print("-----left-----:")
    # print(left_pos, left_quant_xyzw)
    left_ee_pose_quant = np.concatenate([left_pos, left_quant_xyzw])  
    left_ee_pose_quant = np.array(left_ee_pose_quant).reshape(1, 7)   
    left_ee_pose_6d = Transform.convert(               # numpy.ndarray ; dim: 1*9
        tf=left_ee_pose_quant,    
        from_rep="quat",        
        to_rep= "rotation_6d"  
    )
    right_pos = hand_poses.right_pose.pos_xyz           # type: tuple
    right_quant_xyzw = hand_poses.right_pose.quat_xyzw  # type: tuple
    # print("-----right-----:")
    # print(right_pos, right_quant_xyzw)
    right_ee_pose_quant = np.concatenate([right_pos, right_quant_xyzw])  
    right_ee_pose_quant = np.array(right_ee_pose_quant).reshape(1, 7)   
    right_ee_pose_6d = Transform.convert(               # numpy.ndarray ; dim: 1*9 
        tf=right_ee_pose_quant,    
        from_rep="quat",        
        to_rep= "rotation_6d"  
    )

    return left_ee_pose_6d, right_ee_pose_6d

def fk_changed_joint_angle_2_eepose6d(obs_state: np.ndarray, gripper_dim = 1)->np.ndarray: 
    '''
    obs_state: (dim, ) np.array([joint1, joint2, ..., gripper_left, joint1, joint2, ..., gripper_right])
    out_state: (20, )
    '''
    half_dim = obs_state.shape[0] // 2
    end_joint_dim = obs_state.shape[0] - gripper_dim
    left_joint_angles = obs_state[0:(half_dim-gripper_dim)]
    right_joint_angles = obs_state[half_dim:end_joint_dim]
    joint_angles = np.concatenate([left_joint_angles, right_joint_angles]).tolist()
    left_ee_pose_6d, right_ee_pose_6d = joint_angle_2_eep6d(joint_angles)
    left_gripper = obs_state[(half_dim-gripper_dim):half_dim] 
    right_gripper = obs_state[-gripper_dim:]
    out_state = np.concatenate([left_ee_pose_6d.flatten(), left_gripper, 
                                right_ee_pose_6d.flatten(), right_gripper])
    return out_state

def fk_changed_joint_angle_2_eepose6d_rpy(obs_state: np.ndarray, gripper_dim = 1)->np.ndarray: 
    '''
    obs_state: (dim, ) np.array([joint1, joint2, ..., gripper_left, joint1, joint2, ..., gripper_right])
    out_state: (16, )
    '''
    half_dim = obs_state.shape[0] // 2
    end_joint_dim = obs_state.shape[0] - gripper_dim
    left_joint_angles = obs_state[0:(half_dim-gripper_dim)]
    right_joint_angles = obs_state[half_dim:end_joint_dim]
    joint_angles = np.concatenate([left_joint_angles, right_joint_angles]).tolist()
    hand_poses = fk_srv_client(joint_angles)
    left_ee_pos = hand_poses.left_pose.pos_xyz
    left_ee_pose_quat = np.concatenate([left_ee_pos, hand_poses.left_pose.quat_xyzw])
    right_ee_pos = hand_poses.right_pose.pos_xyz
    right_ee_pose_quat = np.concatenate([right_ee_pos, hand_poses.right_pose.quat_xyzw])

    left_ee_pose_rpy = Transform.convert(tf=left_ee_pose_quat, from_rep="quat", to_rep="euler", seq="ZYX", degrees=False)
    right_ee_pose_rpy = Transform.convert(tf=right_ee_pose_quat, from_rep="quat", to_rep="euler", seq="ZYX", degrees=False)

    left_gripper = obs_state[(half_dim-gripper_dim):half_dim] 
    right_gripper = obs_state[-gripper_dim:]
    out_state = np.concatenate([left_ee_pose_rpy, left_gripper])
    out_state = np.concatenate([out_state, right_ee_pose_rpy])
    out_state = np.concatenate([out_state, right_gripper])

    return out_state

if __name__ == "__main__":
    # rospy.init_node("example_fk_srv_node", anonymous=True)
    # # 单位：弧度
    # joint_angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.38, -1.39, -0.29, -0.43, 0.0, -0.17, 0.0]

    # # 调用 FK 正解服务
    # hand_poses = fk_rot6d_srv_client(joint_angles)
    # if hand_poses is not None:
    #     print("left eef position:", hand_poses.left_pose, type(hand_poses.left_pose.pos_xyz))
    #     print("\nright eef position: ", hand_poses.right_pose)
    # else:
    #     print("No hand poses returned")

    rospy.init_node("example_fk_srv_node", anonymous=True)
    rospy.wait_for_service('/ik/fk_srv')

    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.38, -1.39, -0.29, -0.43, 0.0, -0.17, 0.0, 1.0])
    print("state.shape:", state.shape)
    state_6d = fk_changed_joint_angle_2_eepose6d(state)
    print(state_6d, state_6d.shape)

 