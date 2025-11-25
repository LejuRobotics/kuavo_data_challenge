from dataclasses import dataclass
from typing import List, Tuple, Dict
import os
import yaml

@dataclass
class Config_Kuavo_Env:
    # Basic settings
    real: bool
    only_arm: bool
    eef_type: str  # 'qiangnao' or 'leju_claw'
    control_mode: str  # 'joint' or 'eef'
    which_arm: str  # 'left', 'right', or 'both'
    qiangnao_dof_needed: int  # 通常为1，表示只需要第一个关节作为开合依据
    leju_claw_dof_needed: int  # 夹爪需要的自由度数目
    rq2f85_dof_needed: int  # 夹抓rq2f85需要的自由度数目
    is_binary: bool  # 是否使用二值化夹爪，灵巧手，default=false
    
    # Timeline settings
    ros_rate: int

    # Image settings
    image_size: List[int]

    head_init: List[float]|None # 头初始位置，default=None

    # Arm settings
    arm_init: List[float]
    arm_min: List[float]
    arm_max: List[float]
    base_min: List[float]
    base_max: List[float]
    # EEF settings
    eef_min: List[float]
    eef_max: List[float]

    # Input features
    input_images: List[str]
    
    @property
    def use_leju_claw(self) -> bool:
        """Determine if using leju claw based on eef_type."""
        return self.eef_type == 'leju_claw'
    
    @property
    def use_qiangnao(self) -> bool:
        """Determine if using qiangnao based on eef_type."""
        return self.eef_type == 'qiangnao'
    
    @property
    def only_half_up_body(self) -> bool:
        """Always true when only using arm."""
        return True
    
    @property
    def default_camera_names(self) -> List[str]:
        """Get camera names based on which arm is being used."""
        cameras = ['head_cam_h']
        if self.which_arm == 'left':
            cameras.append('wrist_cam_l')
        elif self.which_arm == 'right':
            cameras.append('wrist_cam_r')
        elif self.which_arm == 'both':
            cameras.extend(['wrist_cam_r', 'wrist_cam_l'])
        return cameras
    
    @property
    def slice_robot(self) -> List[Tuple[int, int]]:
        """Get robot slice based on which arm is being used."""
        if self.which_arm == 'left':
            return [(12, 19), (19, 19)]
        elif self.which_arm == 'right':
            return [(12, 12), (19, 26)]
        elif self.which_arm == 'both':
            return [(12, 19), (19, 26)]
        else:
            raise ValueError(f"Invalid which_arm: {self.which_arm}")
    
    @property
    def qiangnao_slice(self) -> List[List[int]]:
        """Get qiangnao slice based on which arm and qiangnao_dof_needed."""
        if self.which_arm == 'left':
            return [[0, self.qiangnao_dof_needed], [6, 6]]  # 左手使用指定自由度，右手不使用
        elif self.which_arm == 'right':
            return [[0, 0], [6, 6 + self.qiangnao_dof_needed]]  # 左手不使用，右手使用指定自由度
        elif self.which_arm == 'both':
            return [[0, self.qiangnao_dof_needed], [6, 6 + self.qiangnao_dof_needed]]  # 双手都使用指定自由度
        else:
            raise ValueError(f"Invalid which_arm: {self.which_arm}")
    
    @property
    def claw_slice(self) -> List[List[int]]:
        """Get claw slice based on which arm."""
        if self.which_arm == 'left':
            return [[0, 1], [1, 1]]  # 左手使用夹爪，右手不使用
        elif self.which_arm == 'right':
            return [[0, 0], [1, 2]]  # 左手不使用，右手使用夹爪
        elif self.which_arm == 'both':
            return [[0, 1], [1, 2]]  # 双手都使用夹爪
        else:
            raise ValueError(f"Invalid which_arm: {self.which_arm}")

def load_kuavo_env_config(config_path: str = None) -> Config_Kuavo_Env:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config YAML file. If None, uses default path.
        
    Returns:
        Config object containing all settings 
    """
    if config_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, 'kuavo_env.yaml')
        
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Validate eef_type
    eef_type = config_dict.get('eef_type')
    if eef_type not in ['qiangnao', 'leju_claw', 'rq2f85']:
        raise ValueError(f"Invalid eef_type: {eef_type}, must be 'qiangnao' or 'leju_claw'")
    
    # Validate which_arm
    which_arm = config_dict.get('which_arm')
    if which_arm not in ['left', 'right', 'both']:
        raise ValueError(f"Invalid which_arm: {which_arm}, must be 'left', 'right', or 'both'")
    
    # Create main Config object
    return Config_Kuavo_Env(
        real=config_dict.get('real', False),
        only_arm=config_dict.get('only_arm', True),
        eef_type=eef_type,
        control_mode=config_dict['control_mode'],
        which_arm=which_arm,
        head_init=config_dict.get('head_init', None),
        qiangnao_dof_needed=config_dict.get('qiangnao_dof_needed', 1),
        leju_claw_dof_needed=config_dict.get('leju_claw_dof_needed', 1),
        rq2f85_dof_needed=config_dict.get('rq2f85_dof_needed', 1),
        ros_rate=config_dict['ros_rate'],
        image_size=config_dict['image_size'],
        arm_init=config_dict['arm_init'],
        arm_min=config_dict['arm_min'],
        arm_max=config_dict['arm_max'],
        base_min=config_dict.get('base_min',None),
        base_max=config_dict.get('base_max',None),
        eef_min=config_dict['eef_min'],
        eef_max=config_dict['eef_max'],
        is_binary=config_dict['is_binary'],
        input_images=config_dict['input_images'],
    )

if __name__ == "__main__":
    # For testing purposes, default config instance
    config = load_kuavo_env_config()
    print(config)
