import yaml
from typing import Tuple
from pathlib import Path


def _find_config_file():
    """find config file, search up to the project root containing the configs directory"""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        config_file = current / "configs" / "platform" / "platform_config.yaml"
        if config_file.exists():
            return config_file
        current = current.parent
    # if not found, use a specific path
    return Path(__file__).parent.parent.parent / "configs" / "platform" / "platform_config.yaml"


_CONFIG_FILE = _find_config_file()
if not _CONFIG_FILE.exists():
    raise FileNotFoundError(f"Platform config not found: {_CONFIG_FILE}")

with open(_CONFIG_FILE, 'r', encoding='utf-8') as f:
    _PLATFORM_CONFIG = yaml.safe_load(f) or {}

if "platforms" not in _PLATFORM_CONFIG:
    raise ValueError(f"Invalid platform config: missing 'platforms' key in {_CONFIG_FILE}")

DEFAULT_PLATFORM = _PLATFORM_CONFIG.get("default", "5w").lower()
_PLATFORMS = _PLATFORM_CONFIG["platforms"]


def _get_config(platform_type: str):
    """get config for specified platform"""
    config = _PLATFORMS.get(platform_type.lower())
    if not config:
        raise ValueError(f"Unsupported platform type: {platform_type}. Supported: {list(_PLATFORMS.keys())}")
    return config


def get_arm_joint_slice(platform_type: str = None) -> Tuple[int, int]:
    """get range of arm joints"""
    c = _get_config(platform_type or DEFAULT_PLATFORM)
    return (c["arm_joint_start"], c["arm_joint_end"])


def get_arm_head_start(platform_type: str = None) -> int:
    """get start index of arm + head joints"""
    return _get_config(platform_type or DEFAULT_PLATFORM)["arm_head_start"]


# export constants for default platform
_default = _get_config(DEFAULT_PLATFORM)
ARM_JOINT_START = _default["arm_joint_start"]
ARM_JOINT_END = _default["arm_joint_end"]
ARM_HEAD_START = _default["arm_head_start"]