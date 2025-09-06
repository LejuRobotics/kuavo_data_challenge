from typing import Any, Dict
from dataclasses import dataclass,fields,field
import copy
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
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
from lerobot.optim.optimizers import AdamConfig,AdamWConfig

T = TypeVar("T", bound="CustomDiffusionConfigWrapper")

@PreTrainedConfig.register_subclass("custom_diffusion")
@dataclass
class CustomDiffusionConfigWrapper(DiffusionConfig):
    custom: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        default_map = {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
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
        # self.input_features = self._normalize_feature_dict(self.input_features)
        # self.output_features = self._normalize_feature_dict(self.output_features)
        self._convert_omegaconf_fields()

    # def _normalize_feature_dict(self, d: Any) -> dict[str, PolicyFeature]:
    #     if isinstance(d, DictConfig):
    #         d = OmegaConf.to_container(d, resolve=True)
    #     if not isinstance(d, dict):
    #         raise TypeError(f"Expected dict or DictConfig, got {type(d)}")

    #     return {
    #         k: PolicyFeature(**v) if isinstance(v, dict) and not isinstance(v, PolicyFeature) else v
    #         for k, v in d.items()
    #     }
    
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

        if self.crop_shape is not None:
            if isinstance(self.crop_shape[0],(list,tuple)):
                (x_start, x_end), (y_start, y_end) = self.crop_shape
                for key, image_ft in self.image_features.items():
                    if x_start < 0 or x_end > image_ft.shape[1] or y_start<0 or y_end > image_ft.shape[2]:
                        raise ValueError(
                            f"`crop_shape` should fit within the images shapes. Got {self.crop_shape} "
                            f"for `crop_shape` and {image_ft.shape} for "
                            f"`{key}`."
                        )
            else:
                for key, image_ft in self.image_features.items():
                    if self.crop_shape[0] > image_ft.shape[1] or self.crop_shape[1] > image_ft.shape[2]:
                        raise ValueError(
                            f"`crop_shape` should fit within the images shapes. Got {self.crop_shape} "
                            f"for `crop_shape` and {image_ft.shape} for "
                            f"`{key}`."
                        )

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
        # 用父类类型调用 from_pretrained，触发 Choice 机制识别子类
        parent_cls = PreTrainedConfig  # 或者直接 DiffusionConfig

        # 调用父类的 from_pretrained，注意传入所有参数和额外参数
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
    
    def get_optimizer_preset(self):
        if self.use_unet:
            print("~~~~~~~~~~~~~~~Use Adam~~~~~~~~~~~~~~~~")
            return AdamConfig(
                lr=self.optimizer_lr,
                betas=self.optimizer_betas,
                eps=self.optimizer_eps,
                weight_decay=self.optimizer_weight_decay,
            )
        else:
            print("~~~~~~~~~~~~~~~Use AdamW~~~~~~~~~~~~~~~~")
            return AdamWConfig(
                lr=self.optimizer_lr,
                betas=self.optimizer_betas,
                eps=self.optimizer_eps,
                weight_decay=self.optimizer_weight_decay,
            )
        