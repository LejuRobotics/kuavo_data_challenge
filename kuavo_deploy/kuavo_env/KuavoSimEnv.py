#!/usr/bin/env python3
import rospy
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Float32, Float32MultiArray,Float64MultiArray,Int32
from std_msgs.msg import Bool
import cv2
import gymnasium as gym
import time
from kuavo_deploy.kuavo_env.KuavoBaseRosEnv import KuavoBaseRosEnv

class KuavoSimEnv(KuavoBaseRosEnv):
    def _reset_eef(self):
        return

    def compute_reward(self):
        return 0