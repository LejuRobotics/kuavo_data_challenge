from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol
import sys, importlib
import numpy as np


lerobot_config_types = importlib.import_module("lerobot.configs.types")
class FeatureType(str, Enum):
    STATE = "STATE"
    VISUAL = "VISUAL"
    ENV = "ENV"
    ACTION = "ACTION"
    REWARD = "REWARD"
    RGB = "RGB"
    DEPTH = "DEPTH"

@dataclass
class PolicyFeature:
    type: FeatureType
    shape: tuple

lerobot_config_types.FeatureType = FeatureType
lerobot_config_types.PolicyFeature = PolicyFeature
sys.modules["lerobot.configs.types"] = lerobot_config_types


from lerobot.datasets.compute_stats import (get_feature_stats, 
                                            sample_indices, 
                                            load_image_as_numpy,
                                            auto_downsample_height_width
                                            )

lerobot_datasets_compute_stats = importlib.import_module("lerobot.datasets.compute_stats")
def custom_sample_images(image_paths: list[str], sampled_indices) -> np.ndarray:

    images = None
    for i, idx in enumerate(sampled_indices):
        path = image_paths[idx]
        # we load as uint8 to reduce memory usage
        img = load_image_as_numpy(path, dtype=np.uint8, channel_first=True)
        img = auto_downsample_height_width(img)

        if images is None:
            images = np.empty((len(sampled_indices), *img.shape), dtype=np.uint8)

        images[i] = img

    return images

def custom_sample_depth(data, sampled_indices):
    """
    Samples depth values from the data based on the sampled indices for the batch dimension.
    
    Parameters:
        data (np.ndarray): A Bx1xHxW array of depth data with dtype np.uint16.
        sampled_indices (np.ndarray): A 1D array containing indices for sampling from the batch.
    
    Returns:
        np.ndarray: A (N, 1, H, W) array containing the sampled depth maps for each sampled index.
    """
    # Initialize a list to store the sampled depth maps
    sampled_depth_maps = []

    # Loop through each index in sampled_indices to get the corresponding depth map
    for idx in sampled_indices:
        # Extract the depth map for the given batch index, maintaining the 1xHxW shape
        sampled_depth_maps.append(auto_downsample_height_width(data[idx, :, :, :]))

    # Convert the list to a numpy array of shape (N, 1, H, W)
    return np.array(sampled_depth_maps)

def compute_episode_stats(episode_data: dict[str, list[str] | np.ndarray], features: dict) -> dict:
    ep_stats = {}
    sampled_indices = sample_indices(len(episode_data["action"]))
    for key, data in episode_data.items():
        if features[key]["dtype"] == "string":
            continue  # HACK: we should receive np.arrays of strings
        elif features[key]["dtype"] in ["image", "video"]:
            ep_ft_array = custom_sample_images(data, sampled_indices)  # data is a list of image paths
            axes_to_reduce = (0, 2, 3)  # keep channel dim
            keepdims = True
        elif features[key]["dtype"] == "uint16":
            ep_ft_array = custom_sample_depth(data, sampled_indices)  # data is already a np.ndarray
            axes_to_reduce = (0, 2, 3)  # compute stats over the first axis
            keepdims = True  # keep as np.array

        else:
            ep_ft_array = data  # data is already a np.ndarray
            axes_to_reduce = 0  # compute stats over the first axis
            keepdims = data.ndim == 1  # keep as np.array
        # print(features[key]["dtype"], ep_ft_array.shape)
        ep_stats[key] = get_feature_stats(ep_ft_array, axis=axes_to_reduce, keepdims=keepdims)
        print("meta stats compute:",key, features[key]["dtype"], ep_stats[key]["count"],ep_stats[key]["mean"].shape,ep_ft_array.shape, )

        # finally, we normalize and remove batch dim for images
        if features[key]["dtype"] in ["image", "video"]:
            ep_stats[key] = {
                k: v if k == "count" else np.squeeze(v / 255.0, axis=0) for k, v in ep_stats[key].items()
            }
        if features[key]["dtype"] == "uint16":
            # ep_stats[key] = {
            #     k: v if k == "count" else np.squeeze(v / 255.0, axis=0) for k, v in ep_stats[key].items()
            # }
            ep_stats[key] = {
                k: v if k == "count" else np.squeeze(v, axis=0) for k, v in ep_stats[key].items()
            }
    return ep_stats

lerobot_datasets_compute_stats.compute_episode_stats = compute_episode_stats
sys.modules["lerobot.datasets.compute_stats"] = lerobot_datasets_compute_stats



lerobot_datasets_utils = importlib.import_module("lerobot.datasets.utils")

def dataset_to_policy_features(features: dict[str, dict]) -> dict[str, PolicyFeature]:
    # TODO(aliberts): Implement "type" in dataset features and simplify this
    # print("~~~~~~~~~features from custom dataset_to_policy_features:~~~~~~~~")
    policy_features = {}
    for key, ft in features.items():
        shape = ft["shape"]
        if ft["dtype"] in ["image", "video", "uint16"]:
            if "depth" in key or "DEPTH" in key:
                type = FeatureType.DEPTH
            else:
                type = FeatureType.VISUAL  # or type = FeatureType.RGB 
            if len(shape) != 3:
                raise ValueError(f"Number of dimensions of {key} != 3 (shape={shape})")

            names = ft["names"]
            # Backward compatibility for "channel" which is an error introduced in LeRobotDataset v2.0 for ported datasets.
            if names[2] in ["channel", "channels"]:  # (h, w, c) -> (c, h, w)
                shape = (shape[2], shape[0], shape[1])
        elif key == "observation.environment_state":
            type = FeatureType.ENV
        elif key.startswith("observation"):
            type = FeatureType.STATE
        elif key == "action":
            type = FeatureType.ACTION
        else:
            continue

        policy_features[key] = PolicyFeature(
            type=type,
            shape=shape,
        )

    return policy_features


lerobot_datasets_utils.dataset_to_policy_features = dataset_to_policy_features
sys.modules["lerobot.datasets.utils"] = lerobot_datasets_utils


# import numpy as np
# import torch
# from torch import Tensor, le, nn
# from lerobot.configs.types import NormalizationMode

# def create_stats_buffers(
#     features: dict[str, PolicyFeature],
#     norm_map: dict[str, NormalizationMode],
#     stats: dict[str, dict[str, Tensor]] | None = None,
# ) -> dict[str, dict[str, nn.ParameterDict]]:
#     """
#     Create buffers per modality (e.g. "observation.image", "action") containing their mean, std, min, max
#     statistics.

#     Args: (see Normalize and Unnormalize)

#     Returns:
#         dict: A dictionary where keys are modalities and values are `nn.ParameterDict` containing
#             `nn.Parameters` set to `requires_grad=False`, suitable to not be updated during backpropagation.
#     """
#     stats_buffers = {}

#     for key, ft in features.items():
#         norm_mode = norm_map.get(ft.type, NormalizationMode.IDENTITY)
#         if norm_mode is NormalizationMode.IDENTITY:
#             continue

#         assert isinstance(norm_mode, NormalizationMode)

#         shape = tuple(ft.shape)

#         if ft.type is FeatureType.VISUAL or ft.type is FeatureType.RGB or ft.type is FeatureType.DEPTH:
#             # sanity checks
#             assert len(shape) == 3, f"number of dimensions of {key} != 3 ({shape=}"
#             c, h, w = shape
#             assert c < h and c < w, f"{key} is not channel first ({shape=})"
#             # override image shape to be invariant to height and width
#             shape = (c, 1, 1)

#         # Note: we initialize mean, std, min, max to infinity. They should be overwritten
#         # downstream by `stats` or `policy.load_state_dict`, as expected. During forward,
#         # we assert they are not infinity anymore.

#         buffer = {}
#         if norm_mode is NormalizationMode.MEAN_STD:
#             mean = torch.ones(shape, dtype=torch.float32) * torch.inf
#             std = torch.ones(shape, dtype=torch.float32) * torch.inf
#             buffer = nn.ParameterDict(
#                 {
#                     "mean": nn.Parameter(mean, requires_grad=False),
#                     "std": nn.Parameter(std, requires_grad=False),
#                 }
#             )
#         elif norm_mode is NormalizationMode.MIN_MAX:
#             min = torch.ones(shape, dtype=torch.float32) * torch.inf
#             max = torch.ones(shape, dtype=torch.float32) * torch.inf
#             buffer = nn.ParameterDict(
#                 {
#                     "min": nn.Parameter(min, requires_grad=False),
#                     "max": nn.Parameter(max, requires_grad=False),
#                 }
#             )

#         # TODO(aliberts, rcadene): harmonize this to only use one framework (np or torch)
#         if stats:
#             if isinstance(stats[key]["mean"], np.ndarray):
#                 if norm_mode is NormalizationMode.MEAN_STD:
#                     buffer["mean"].data = torch.from_numpy(stats[key]["mean"]).to(dtype=torch.float32)
#                     buffer["std"].data = torch.from_numpy(stats[key]["std"]).to(dtype=torch.float32)
#                 elif norm_mode is NormalizationMode.MIN_MAX:
#                     buffer["min"].data = torch.from_numpy(stats[key]["min"]).to(dtype=torch.float32)
#                     buffer["max"].data = torch.from_numpy(stats[key]["max"]).to(dtype=torch.float32)
#             elif isinstance(stats[key]["mean"], torch.Tensor):
#                 # Note: The clone is needed to make sure that the logic in save_pretrained doesn't see duplicated
#                 # tensors anywhere (for example, when we use the same stats for normalization and
#                 # unnormalization). See the logic here
#                 # https://github.com/huggingface/safetensors/blob/079781fd0dc455ba0fe851e2b4507c33d0c0d407/bindings/python/py_src/safetensors/torch.py#L97.
#                 if norm_mode is NormalizationMode.MEAN_STD:
#                     buffer["mean"].data = stats[key]["mean"].clone().to(dtype=torch.float32)
#                     buffer["std"].data = stats[key]["std"].clone().to(dtype=torch.float32)
#                 elif norm_mode is NormalizationMode.MIN_MAX:
#                     buffer["min"].data = stats[key]["min"].clone().to(dtype=torch.float32)
#                     buffer["max"].data = stats[key]["max"].clone().to(dtype=torch.float32)
#             else:
#                 type_ = type(stats[key]["mean"])
#                 raise ValueError(f"np.ndarray or torch.Tensor expected, but type is '{type_}' instead.")

#         stats_buffers[key] = buffer
#     return stats_buffers

# lerobot_policies_normalize = importlib.import_module("lerobot.policies.normalize")
# lerobot_policies_normalize.create_stats_buffers = create_stats_buffers
# sys.modules["lerobot.policies.normalize"] = lerobot_policies_normalize




