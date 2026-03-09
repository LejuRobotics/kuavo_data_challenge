"""
PI0 Configuration Wrapper

This module provides a custom wrapper for PI0Config that adds support for depth image features
and RGB-Depth fusion, similar to the ACT and Diffusion policy wrappers.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, fields, field
import copy
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from omegaconf import DictConfig, OmegaConf, ListConfig
from copy import deepcopy
from pathlib import Path
import draccus
from huggingface_hub.constants import CONFIG_NAME
from typing import TypeVar

T = TypeVar("T", bound="CustomPI0ConfigWrapper")


@PreTrainedConfig.register_subclass("custom_pi0")
@dataclass
class CustomPI0ConfigWrapper(PI0Config):
    """
    Custom PI0 Configuration Wrapper
    
    Extends PI0Config to support depth image features and RGB-Depth fusion.
    This wrapper allows passing depth-related parameters that are not part of
    the base PI0Config class.
    
    Attributes:
        custom: Dictionary for storing custom configuration parameters
        use_depth: Whether to use depth images (default: False)
        depth_features: List of depth feature keys (e.g., ["observation.depth_h"])
        depth_fusion_method: Method for fusing RGB and depth features
        depth_fusion_dim: Dimension for depth fusion (None = auto)
        depth_backbone: Backbone model for depth feature extraction
        depth_preprocessing: Preprocessing method for depth images
        depth_scale: Scaling factor for depth values (uint16 to meters)
    """
    
    # Custom dictionary for storing additional parameters
    custom: Dict[str, Any] = field(default_factory=dict)


    def __post_init__(self):
        """
        Initialize the configuration wrapper.
        
        This method:
        1. Calls the parent class __post_init__
        2. Merges normalization mappings
        3. Extracts custom parameters from the 'custom' dict and sets them as attributes
        4. Converts OmegaConf types to native Python types
        """
        super().__post_init__()
        
        # Merge normalization mappings with defaults
        default_map = {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
        
        merged = copy.deepcopy(default_map)
        merged.update(self.normalization_mapping)
        self.normalization_mapping = merged
        
        # Extract custom settings from 'custom' dict and set them as attributes
        # This allows passing depth parameters through the 'custom' dict if needed
        if isinstance(self.custom, DictConfig) or isinstance(self.custom, dict):
            for k, v in self.custom.items():
                if hasattr(self, k):
                    print(f"\033[93mCustom setting '{k}: {v}' conflicts with the parent base configuration. The custom value will override the base setting.\033[0m")  # Remove it from 'custom' and modify in the parent configuration instead.
                setattr(self, k, v)
        # Convert OmegaConf types to native Python types
        self._convert_omegaconf_fields()

    def _convert_omegaconf_fields(self):
        """
        Convert OmegaConf DictConfig and ListConfig to native Python types.
        
        This is necessary because OmegaConf types can cause issues with dataclass
        operations and serialization.
        """
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, (ListConfig, DictConfig)):
                converted = OmegaConf.to_container(val, resolve=True)
                setattr(self, f.name, converted)

    @property
    def image_features(self) -> dict[str, PolicyFeature]:
        """
        Get all image (RGB/VISUAL) features from input features.
        
        Returns:
            Dictionary mapping feature keys to PolicyFeature objects for RGB/VISUAL features
        """
        return {
            key: ft 
            for key, ft in self.input_features.items() 
            if (ft.type is FeatureType.RGB) or (ft.type is FeatureType.VISUAL)
        }

    @property
    def depth_feature(self) -> dict[str, PolicyFeature]:
        """
        Get all depth features from input features as a dictionary.
        
        This property extracts depth features from input_features based on FeatureType.DEPTH.
        Use the 'depth_features' field (list of strings) to specify which depth features to use.
        
        Returns:
            Dictionary mapping feature keys to PolicyFeature objects for DEPTH features
        """
        return {
            key: ft 
            for key, ft in self.input_features.items() 
            if ft.type is FeatureType.DEPTH
        }

    def validate_features(self) -> None:
        """
        Validate input features.
        
        This method:
        1. Calls the parent class validate_features
        2. Validates that all image features have the same shape
        3. Validates that all depth features have the same shape (if depth is enabled)
        """
        # Call parent validation first
        super().validate_features()
        
        # Validate image features have consistent shapes
        if len(self.image_features) > 0:
            first_image_key, first_image_ft = next(iter(self.image_features.items()))
            for key, image_ft in self.image_features.items():
                if image_ft.shape != first_image_ft.shape:
                    raise ValueError(
                        f"`{key}` does not match `{first_image_key}`, "
                        f"but we expect all image shapes to match."
                    )
        
        # Validate depth features have consistent shapes (if depth is enabled)
        if self.use_depth and len(self.depth_feature) > 0:
            first_depth_key, first_depth_ft = next(iter(self.depth_feature.items()))
            for key, depth_ft in self.depth_feature.items():
                if depth_ft.shape != first_depth_ft.shape:
                    raise ValueError(
                        f"`{key}` does not match `{first_depth_key}`, "
                        f"but we expect all depth shapes to match."
                    )
        elif self.use_depth and len(self.depth_feature) == 0:
            print("Warning: use_depth is True but no depth features found in input_features!")

    def _save_pretrained(self, save_directory: Path) -> None:
        """
        Save the configuration to a file.
        
        This method saves the configuration while handling custom parameters
        that were extracted from the 'custom' dict.
        
        Args:
            save_directory: Directory where the config file should be saved
        """
        cfg_copy = deepcopy(self)
        
        # Move custom attributes back to 'custom' dict for saving
        if isinstance(cfg_copy.custom, dict):
            for k in list(cfg_copy.custom.keys()):
                if hasattr(cfg_copy, k):
                    delattr(cfg_copy, k)
        elif hasattr(cfg_copy, "custom") and hasattr(cfg_copy.custom, "keys"):
            for k in list(cfg_copy.custom.keys()):
                if hasattr(cfg_copy, k):
                    delattr(cfg_copy, k)
        
        # Save using draccus
        with open(save_directory / CONFIG_NAME, "w") as f, draccus.config_type("json"):
            draccus.dump(cfg_copy, f, indent=4)

    @classmethod
    def from_pretrained(
        cls: type[T],
        pretrained_name_or_path: str | Path,
        *,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **policy_kwargs,
    ) -> T:
        """
        Load a pretrained configuration.
        
        This method delegates to the parent class's from_pretrained method
        to handle loading from HuggingFace Hub or local paths.
        
        Args:
            pretrained_name_or_path: Name or path of the pretrained config
            **kwargs: Additional arguments for loading
            
        Returns:
            Instance of CustomPI0ConfigWrapper loaded from pretrained config
        """
        parent_cls = PreTrainedConfig
        return parent_cls.from_pretrained(
            pretrained_name_or_path,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
            **policy_kwargs,
        )

