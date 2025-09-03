from dataclasses import dataclass
from typing import List, Tuple, Dict
import os
import yaml

@dataclass
class Config_Inference:
    # Basic settings
    go_bag_path: str

    policy_type: str
    use_delta: bool
    eval_episodes: int
    seed: int
    start_seed: int
    device: str
    task: str
    method: str
    timestamp: str
    epoch: int
    max_episode_steps: int
    env_name: str

def load_inference_config(config_path: str = None) -> Config_Inference:
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
    
    # Create main Config object
    return Config_Inference(
        go_bag_path=config_dict['go_bag_path'],
        policy_type=config_dict['policy_type'],
        use_delta=config_dict['use_delta'],
        eval_episodes=config_dict['eval_episodes'],
        seed=config_dict['seed'],
        start_seed=config_dict['start_seed'],
        device=config_dict['device'],
        task=config_dict['task'],
        method=config_dict['method'],
        timestamp=config_dict['timestamp'],
        epoch=config_dict['epoch'],
        max_episode_steps=config_dict['max_episode_steps'],
        env_name=config_dict['env_name']
    )

if __name__ == "__main__":
    # For testing purposes, default config instance
    config = load_inference_config()
    print(config)
