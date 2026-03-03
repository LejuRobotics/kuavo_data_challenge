"""
Single merged configuration loader for Kuavo.
Loads ./configs/deploy/kuavo_env.yaml by default (keeps original YAML structure).
Provides:
  - ConfigEnv (environment) with slice properties:
      .slice_robot, .qiangnao_slice, .claw_slice
  - ConfigInference (inference)
  - KuavoConfig (master)
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Any, Dict
import os
import yaml


@dataclass
class Range:
    min: List[float]
    max: List[float]

@dataclass
class LimitsConfig:
    joint_q: Range = field(default_factory=lambda: Range([-3.14]*14, [3.14]*14))
    gripper: Range = field(default_factory=lambda: Range([0, 0], [1, 1]))
    eef: Range = field(default_factory=lambda: Range(
        [-1, -1, -1, -3.14, -3.14, -3.14,
         -1, -1, -1, -3.14, -3.14, -3.14],
        [1, 1, 1, 3.14, 3.14, 3.14,
         1, 1, 1, 3.14, 3.14, 3.14]
    ))
    eef_relative: Range = field(default_factory=lambda: Range(
        [-0.005, -0.0075, -0.004, -0.03, -0.03, -0.05,
         -0.005, -0.0075, -0.004, -0.03, -0.03, -0.05],
        [0.005, 0.0075, 0.004, 0.03, 0.03, 0.05,
         0.005, 0.0075, 0.004, 0.03, 0.03, 0.05]
    ))
    base: Range = field(default_factory=lambda: Range([-2.0, -2.0, -3.14, 0],
                                                      [2.0, 2.0, 3.14, 1]))

# -----------------------
# Environment Dataclass
# -----------------------
@dataclass
class ConfigEnv:
    env_name: str = "Kuavo-Sim"
    real: bool = False
    only_arm: bool = True
    eef_type: str = "rq2f85"
    control_mode: str = "joint"
    which_arm: str = "both"
    head_init: Optional[List[float]] = field(default_factory=lambda: [0.0, 0.0])
    use_delta: bool = False
    delta_type: str = "Tsub"  # "Tsub","Tinv","RPY"
    ros_rate: int = 10
    image_size: List[int] = field(default_factory=lambda: [640, 480])
    depth_range: List[int] = field(default_factory=lambda: [0, 1500])
    obs_key_map: Dict[str, List[Any]] = field(default_factory=dict)
    arm_state_keys: List[str]=field(default_factory=list)
    ratio: float = 0.5
    frame_alignment: bool = True
    qiangnao_dof_needed: int = 1

    fk_joint_angles_for_reset: Optional[List[float]] = None
    rotation_threshold: Optional[float] = None
    
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    is_binary: bool = False

    # -------- Validation ----------
    def validate(self):
        if self.eef_type not in ["rq2f85", "leju_claw", "qiangnao"]:
            raise ValueError(f"Invalid eef_type: {self.eef_type}. Valid: rq2f85, leju_claw, qiangnao")
        if self.which_arm not in ["left", "right", "both"]:
            raise ValueError(f"Invalid which_arm: {self.which_arm}. Valid: left, right, both")
        if not isinstance(self.image_size, list) or len(self.image_size) != 2:
            raise ValueError("image_size must be a list [height, width]")
        # ensure lists lengths for arm bounds
        if not (len(self.limits["joint_q"]["max"]) == len(self.limits["joint_q"]["min"]) == 14):
            raise ValueError("Robot arm_min/arm_max must be lists of length 14")
        if self.qiangnao_dof_needed != 1: # not in [1, 7]:
            raise ValueError("qiangnao_dof_needed must be 1 now!")
            # raise ValueError("qiangnao_dof_needed must be either 1 or 7")

    # -------- Derived properties ----------
    @property
    def joint_q_slice(self):
        return {
            "left": [[12, 19]],
            "right": [[19, 26]],
            "both": [[12, 19], [19, 26]]
        }[self.which_arm]

    @property
    def gripper_slice(self):
        if self.eef_type == "rq2f85" or self.eef_type == "leju_claw":
            return {
                "left": [[0, 1]],
                "right": [[1, 2]],
                "both": [[0, 1], [1, 2]]
            }[self.which_arm]
        elif self.eef_type == "qiangnao" and self.qiangnao_dof_needed == 1:
            return {
                "left": [[0, 1]],
                "right": [[6, 7]],
                "both": [[0, 1], [6, 7]]
            }[self.which_arm]
        else:
            raise ValueError("Unsupported eef_type or dof config")

    # ---------------- obs_key_map build ----------------
    def build_obs_key_map(self) -> Dict[str, Any]:
        obs_map = {}
        for key, info in self.obs_key_map.items():
            base = {
                "topic": info[0],
                "msg_type": info[1],
                "frequency": info[2],
                "handle": {"params": {}}
            }
            # 统一规则化参数处理
            if len(info) == 4 and isinstance(info[3], list):
                base["handle"]["params"]["resize_wh"] = info[3]
            if len(info) == 5 and isinstance(info[3], list) and isinstance(info[4], list):
                base["handle"]["params"]["resize_wh"] = info[3]
                base["handle"]["params"]["depth_range"] = info[4]

            # 特殊键处理
            if key == "joint_q":
                base["handle"]["params"]["slice"] = self.joint_q_slice
            if key in ["rq2f85", "qiangnao", "leju_claw"]:
                base["handle"]["params"]["slice"] = self.gripper_slice
                obs_map["gripper"] = base
                continue
            if key == "eef_pose" and len(info) >= 3 and info[0] == "computed":
                obs_map["eef_pose"] = {
                    "type": "computed",
                    "source": info[1],
                    "frequency": info[2]
                }
                continue
            obs_map[key] = base
        return obs_map



# -----------------------
# Inference Dataclass
# -----------------------
@dataclass
class ConfigInference:
    go_bag_path: str = ""
    policy_type: str = "diffusion"  # 支持 diffusion, act 等
    eval_episodes: int = 1
    seed: int = 42
    start_seed: int = 42
    device: str = "cuda"  # or "cpu"
    task: str = ""
    method: str = ""
    timestamp: str = ""
    epoch: int = 1
    max_episode_steps: int = 1000
    env_name: str = "Kuavo-Sim"

    def validate(self):
        if self.policy_type not in ["diffusion", "act"]:
            # 若将来支持更多策略，请在此扩展
            raise ValueError(f"Unsupported policy_type '{self.policy_type}'")
        if self.device not in ["cuda", "cpu"]:
            raise ValueError("device must be 'cuda' or 'cpu'")


# -----------------------
# Master config
# -----------------------
@dataclass
class KuavoConfig:
    env: ConfigEnv
    inference: ConfigInference

    def validate(self):
        self.env.validate()
        self.inference.validate()


# -----------------------
# Loader
# -----------------------
def load_kuavo_config(config_path: Optional[str] = None) -> KuavoConfig:
    """
    Load config from YAML.
    Default path: ./configs/deploy/kuavo_env.yaml (same name as your original file)
    """
    if config_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "../configs", "deploy", "kuavo_env.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # The user's original YAML was mostly top-level keys (not nested under env/inference).
    # We'll support both styles:
    #  - top-level flat (as your original): keys like 'real', 'policy_type', ...
    #  - nested style: {env: {...}, inference: {...}}
    if 'env' in cfg and 'inference' in cfg:
        env_cfg: Dict[str, Any] = cfg.get('env', {})
        inf_cfg: Dict[str, Any] = cfg.get('inference', {})
    else:
        # 自动根据 dataclass 字段划分 env / inference
        env_fields = set(ConfigEnv.__dataclass_fields__.keys())
        inf_fields = set(ConfigInference.__dataclass_fields__.keys())

        env_cfg = {}
        inf_cfg = {}

        for k, v in cfg.items():
            if k in env_fields:
                env_cfg[k] = v
            elif k in inf_fields:
                inf_cfg[k] = v
            elif k == "limits" and isinstance(v, dict):
                def dict_to_range(d):
                    return Range(d.get("min", []), d.get("max", []))

                env_cfg["limits"] = LimitsConfig(
                    joint_q=dict_to_range(v.get("joint_q", {})),
                    gripper=dict_to_range(v.get("gripper", {})),
                    eef=dict_to_range(v.get("eef", {})),
                    eef_relative=dict_to_range(v.get("eef_relative", {})),
                    base=dict_to_range(v.get("base", {})),
                )
            else:
                env_cfg[k] = v
    # Merge defaults with provided config
    default_env = ConfigEnv()
    default_inf = ConfigInference()

    merged_env = {**asdict(default_env), **env_cfg}
    merged_inf = {**asdict(default_inf), **inf_cfg}

    env = ConfigEnv(**merged_env)
    inference = ConfigInference(**merged_inf)

    config = KuavoConfig(env=env, inference=inference)
    config.env.obs_key_map = config.env.build_obs_key_map()
    config.validate()
    return config


# -----------------------
# Quick test when run as script
# -----------------------
if __name__ == "__main__":
    cfg = load_kuavo_config()
    print(isinstance(cfg, KuavoConfig))
    print("=== Env basic ===")
    print("eef_type:", cfg.env.eef_type)
    print("eef_name:", cfg.env.env_name)
    print("which_arm:", cfg.env.which_arm)
    print("cam keys:", cfg.env.obs_key_map)
    print("slice_robot:", cfg.env.gripper_slice)
    print("qiangnao_slice:", cfg.env.joint_q_slice)
    print("=== Inference basic ===")
    print("policy_type:", cfg.inference.policy_type)
    print("device:", cfg.inference.device)
    print("arm_state_keys",cfg.env.arm_state_keys)
