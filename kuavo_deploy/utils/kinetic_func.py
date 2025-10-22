from kuavo_deploy.utils.logging_utils import setup_logger
import rospy

from kuavo_humanoid_sdk.msg.kuavo_msgs.srv import fkSrv, twoArmHandPoseCmdSrv
import numpy as np

log_robot = setup_logger("robot")

# ================ 独立的计算函数 ================

def fk_compute_func(joint_angles):
    """
    前向运动学计算函数
    通过 ROS service 调用 FK 服务，计算末端执行器姿态
    
    Args:
        joint_angles: 关节角度列表
        
    Returns:
        FK service 返回的 hand_poses
    """
    rospy.wait_for_service('/ik/fk_srv')
    try:
        fk_srv = rospy.ServiceProxy('/ik/fk_srv', fkSrv)
        fk_result = fk_srv(joint_angles)
        log_robot.debug(f"FK result success: {fk_result.success}")
        return fk_result.hand_poses
    except rospy.ServiceException as e:
        log_robot.error(f"FK Service call failed: {e}")
        return None


def ik_compute_func(eef_pose_msg):
    """
    逆向运动学计算函数
    通过 ROS service 调用 IK 服务，计算关节角度
    
    Args:
        eef_pose_msg: 末端执行器姿态消息
        
    Returns:
        关节角度数组，或 False（失败时）
    """
    rospy.wait_for_service('/ik/two_arm_hand_pose_cmd_srv')
    try:
        ik_srv = rospy.ServiceProxy('/ik/two_arm_hand_pose_cmd_srv', twoArmHandPoseCmdSrv)
        res = ik_srv(eef_pose_msg)
        return np.concatenate((res.hand_poses.left_pose.joint_angles, res.hand_poses.right_pose.joint_angles), axis=0)
    except rospy.ServiceException as e:
        log_robot.error(f"IK Service call failed: {e}")
        return False