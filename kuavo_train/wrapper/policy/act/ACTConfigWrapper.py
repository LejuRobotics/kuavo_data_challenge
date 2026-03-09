from typing import Any, Dict
from dataclasses import dataclass,fields,field
import copy
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from omegaconf import DictConfig, OmegaConf, ListConfig
from copy import deepcopy
from pathlib import Path
import draccus
from huggingface_hub.constants import CONFIG_NAME
import os, builtins,json,tempfile
from pathlib import Path
from typing import TypeVar
from huggingface_hub import HfApi, ModelCard, ModelCardData, hf_hub_download
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from huggingface_hub.errors import HfHubHTTPError

T = TypeVar("T", bound="CustomACTConfigWrapper")

@PreTrainedConfig.register_subclass("custom_act")
@dataclass
class CustomACTConfigWrapper(ACTConfig):
    custom: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        default_map = {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }

        # merge and update the normalization_mapping
        merged = copy.deepcopy(default_map)
        merged.update(self.normalization_mapping)

        self.normalization_mapping = merged
        # make custom settings in main config for better save
        if isinstance(self.custom, DictConfig) or isinstance(self.custom, dict):
            for k, v in self.custom.items():
                if not hasattr(self, k):
                    # print("from config",k,v)
                    setattr(self, k, v)
                else:
                    raise ValueError(f"Custom setting '{k}: {v}' conflicts with the parent base configuration. Remove it from 'custom' and modify in the parent configuration instead.")
        self._convert_omegaconf_fields()
    
    def _convert_omegaconf_fields(self):
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, (ListConfig, DictConfig)):
                converted = OmegaConf.to_container(val, resolve=True)
                setattr(self, f.name, converted)

    @property
    def image_features(self) -> dict[str, PolicyFeature]:
        return {key: ft for key, ft in self.input_features.items() if (ft.type is FeatureType.RGB) or (ft.type is FeatureType.VISUAL)}
    
    @property
    def depth_features(self) -> dict[str, PolicyFeature]:
        return {key: ft for key, ft in self.input_features.items() if ft.type is FeatureType.DEPTH}
    

    def validate_features(self) -> None:
        if len(self.image_features) == 0 and self.env_state_feature is None:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

        # Check that all input images have the same shape.
        first_image_key, first_image_ft = next(iter(self.image_features.items()))
        
        for key, image_ft in self.image_features.items():
            if image_ft.shape != first_image_ft.shape:
                raise ValueError(
                    f"`{key}` does not match `{first_image_key}`, but we expect all image shapes to match."
                )
        if len(self.depth_features)==0:
            print("No depth features found!")
        else:
            first_depth_key, first_depth_ft = next(iter(self.depth_features.items()))
            for key, image_ft in self.depth_features.items():
                if image_ft.shape != first_depth_ft.shape:
                    raise ValueError(
                        f"`{key}` does not match `{first_depth_key}`, but we expect all image shapes to match."
                    )
            
    def _save_pretrained(self, save_directory: Path) -> None:
        cfg_copy = deepcopy(self)
        if isinstance(cfg_copy.custom, dict):
            for k in list(cfg_copy.custom.keys()):
                if hasattr(cfg_copy, k):
                    delattr(cfg_copy, k)
        elif hasattr(cfg_copy, "custom") and hasattr(cfg_copy.custom, "keys"):
            for k in list(cfg_copy.custom.keys()):
                if hasattr(cfg_copy, k):
                    delattr(cfg_copy, k)
        with open(save_directory / CONFIG_NAME, "w") as f, draccus.config_type("json"):
            draccus.dump(cfg_copy, f, indent=4)

    
    @classmethod
    def from_pretrained(
        cls: type[T],
        pretrained_name_or_path: str | Path,
        *,
        force_download: bool = False,
        resume_download: bool = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **policy_kwargs,
    ) -> T:
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
        