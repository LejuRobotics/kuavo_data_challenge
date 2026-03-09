import lerobot_patches.custom_patches  # Ensure custom patches are applied, DON'T REMOVE THIS LINE!
from lerobot.configs.policies import PolicyFeature
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
from pathlib import Path
from functools import partial

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import shutil
import accelerate
from accelerate.utils import DistributedDataParallelKwargs
from hydra.utils import instantiate
# from diffusers.optimization import get_scheduler

from lerobot.configs.types import FeatureType, NormalizationMode
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata, LeRobotDataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.utils.random_utils import set_seed
from lerobot.policies.factory import make_pre_post_processors
from kuavo_train.wrapper.policy.diffusion.DiffusionPolicyWrapper import CustomDiffusionPolicyWrapper
from kuavo_train.wrapper.policy.act.ACTPolicyWrapper import CustomACTPolicyWrapper
from kuavo_train.wrapper.dataset.LeRobotDatasetWrapper import CustomLeRobotDataset
from kuavo_train.utils.augmenter import crop_image, resize_image, DeterministicAugmenterColor
from kuavo_train.utils.utils import save_rng_state, load_rng_state
from lerobot.policies.act.modeling_act import ACTPolicy
from diffusers.optimization import get_scheduler
from utils.transforms import ImageTransforms, ImageTransformsConfig, ImageTransformConfig

from functools import partial
from contextlib import nullcontext
from lerobot.processor import ProcessorStep, NormalizerProcessorStep
from lerobot.processor.core import TransitionKey
from lerobot.configs.types import PipelineFeatureType, PolicyFeature


def build_augmenter(cfg):
    """Since operations such as cropping and resizing in LeRobot are implemented at the model level 
    rather than at the data level, we provide only RGB image augmentations on the data side here, 
    with support for customization. For more details, refer to configs/policy/diffusion_config.yaml. 
    To define custom transformations, please see utils.transforms.py."""

    img_tf_cfg = ImageTransformsConfig(
        enable=cfg.get("enable", False),
        max_num_transforms=cfg.get("max_num_transforms", 3),
        random_order=cfg.get("random_order", False),
        tfs={}
    )

    # deal tfs part
    if "tfs" in cfg:
        for name, tf_dict in cfg["tfs"].items():
            img_tf_cfg.tfs[name] = ImageTransformConfig(
                weight=tf_dict.get("weight", 1.0),
                type=tf_dict.get("type", "Identity"),
                kwargs=tf_dict.get("kwargs", {}),
            )
    return ImageTransforms(img_tf_cfg)


def build_delta_timestamps(dataset_metadata, policy_cfg):
    """Build delta timestamps for observations and actions."""
    obs_indices = getattr(policy_cfg, "observation_delta_indices", None)
    act_indices = getattr(policy_cfg, "action_delta_indices", None)
    if obs_indices is None and act_indices is None:
        return None

    delta_timestamps = {}
    for key in dataset_metadata.info["features"]:
        if "observation" in key and obs_indices is not None:
            delta_timestamps[key] = [i / dataset_metadata.fps for i in obs_indices]
        elif "action" in key and act_indices is not None:
            delta_timestamps[key] = [i / dataset_metadata.fps for i in act_indices]

    return delta_timestamps if delta_timestamps else None


def build_optimizer_and_scheduler(policy, cfg, total_frames, accelerator):
    """Return optimizer and scheduler."""
    optimizer = policy.config.get_optimizer_preset().build(policy.parameters())
    # If `max_training_step` is specified, it takes precedence; 
    # otherwise, the value is automatically determined based on `max_epoch`.
    if cfg.training.max_training_step is None:
        effective_batch_size = cfg.training.batch_size * accelerator.num_processes
        # updates_per_epoch = (total_frames // (cfg.training.batch_size * cfg.training.accumulation_steps)) + 1
        updates_per_epoch = max(1, total_frames // (effective_batch_size * cfg.training.accumulation_steps))
        num_training_steps = cfg.training.max_epoch * updates_per_epoch
    else:
        num_training_steps = cfg.training.max_training_step
    lr_scheduler = policy.config.get_scheduler_preset()
    if lr_scheduler is not None:
        lr_scheduler = lr_scheduler.build(optimizer, num_training_steps)
    else:
        lr_scheduler = get_scheduler(
            name=cfg.training.scheduler_name,
            optimizer=optimizer,
            num_warmup_steps=cfg.training.scheduler_warmup_steps,
            num_training_steps=num_training_steps,
        )

    # or you can set your optimizer and lr_scheduler here and replace it.
    return optimizer, lr_scheduler

def build_policy(name, policy_cfg):
    policy = {
        "diffusion": CustomDiffusionPolicyWrapper,
        "act": CustomACTPolicyWrapper,
    }[name](policy_cfg)
    return policy

def build_policy_config(cfg, input_features, output_features):
    def _normalize_feature_dict(d: Any) -> dict[str, PolicyFeature]:
        if isinstance(d, DictConfig):
            d = OmegaConf.to_container(d, resolve=True)
        if not isinstance(d, dict):
            raise TypeError(f"Expected dict or DictConfig, got {type(d)}")

        return {
            k: PolicyFeature(**v) if isinstance(v, dict) and not isinstance(v, PolicyFeature) else v
            for k, v in d.items()
        }

    policy_cfg = instantiate(
        cfg.policy,
        input_features=input_features,
        output_features=output_features,
        device=cfg.training.device,
    )
                
    policy_cfg.input_features = _normalize_feature_dict(policy_cfg.input_features)
    policy_cfg.output_features = _normalize_feature_dict(policy_cfg.output_features)
    return policy_cfg

class AugmentationProcessorStep(ProcessorStep):
    def __init__(self, transform, cam_keys):
        super().__init__()
        self.transform = transform
        self.cam_keys = [k for k in cam_keys if "depth" not in k]  # list of keys in the transition dict to augment

    def __call__(self, transition):
        # Store the current transition (required by ProcessorStep)
        new_transition = transition.copy()

        # Apply transform to each camera key
        data_dict = new_transition.get(TransitionKey.OBSERVATION)
        if data_dict is not None:
            # new_data_dict = {
            #     k: self.transform(v) if k in self.cam_keys else v
            #     for k, v in data_dict.items()
            # }
            new_data_dict = {}
            for k, v in data_dict.items():
                
                if k in self.cam_keys:
                    # print(k)
                    new_data_dict[k] = self.transform(v)
                else:
                    new_data_dict[k] = v
            # print(new_data_dict['observation.images.head_cam_h'].device)
            new_transition[TransitionKey.OBSERVATION] = new_data_dict
            return new_transition
        else:
            return new_transition
        

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Returns the input features unchanged.

        Device and dtype transformations do not alter the fundamental definition of the features (e.g., shape).

        Args:
            features: A dictionary of policy features.

        Returns:
            The original dictionary of policy features.
        """
        return features
    

def insert_before_normalizer(pipeline, new_step):
    """
    Insert a processor step before the first NormalizerProcessorStep.
    If no NormalizerProcessorStep is found, append at the end.
    """
    for i, step in enumerate(pipeline.steps):
        if isinstance(step, NormalizerProcessorStep):
            pipeline.steps.insert(i, new_step)
            print(f"Inserted {new_step.__class__.__name__} before NormalizerProcessorStep", {i})
            return new_step
    pipeline.steps.append(new_step)
    print(f"No NormalizerProcessorStep found, appended {new_step.__class__.__name__} at the end")
    return new_step

def remove_aug_step(pipeline, step_to_remove):
    """
    Remove the given step from the pipeline if it exists.
    """
    if step_to_remove in pipeline.steps:
        pipeline.steps.remove(step_to_remove)
        print(f"Removed {step_to_remove.__class__.__name__}")
    else:
        print(f"Step {step_to_remove.__class__.__name__} not found in pipeline")

@hydra.main(config_path="../configs/policy/", config_name="diffusion_config", version_base=None)
def main(cfg: DictConfig):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    # Initialize Accelerator
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=cfg.training.accumulation_steps,
        log_with=None,                        # Disable logging
        # log_with="tensorboard",             # Disable logging
        device_placement=True,                # Explicitly enable device placement
        step_scheduler_with_optimizer=False,  # A fix to the stepping logic as accelerate might make this thread-unsafe.
        mixed_precision="fp16" if cfg.policy.get("use_amp", False) else "no",
        kwargs_handlers=[ddp_kwargs]          # transfer DDP kwargs
    )

    # With Accelerate, we get the device from accelerator
    device = accelerator.device

    # set_seed(cfg.training.seed)
    accelerate.utils.set_seed(cfg.training.seed)

    # mkdir and output TensorBoard only in the main process
    output_directory = None
    if accelerator.is_main_process:
        output_directory = Path(cfg.training.output_directory) / f"run_{cfg.timestamp}"
        output_directory.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(output_directory))

    # Dataset metadata and features
    dataset_metadata = LeRobotDatasetMetadata(cfg.repoid, root=cfg.root)
    features = dataset_to_policy_features(dataset_metadata.features)
    input_features = {k: ft for k, ft in features.items() if ft.type is not FeatureType.ACTION}
    output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}

    # instantiate the policy
    policy_cfg = build_policy_config(cfg, input_features, output_features)
    # Build policy
    policy = build_policy(cfg.policy_name, policy_cfg)
    accelerator.wait_for_everyone()
    preprocessor, postprocessor = make_pre_post_processors(policy_cfg, dataset_stats=dataset_metadata.stats)
    if accelerator.is_main_process:
        preprocessor.save_pretrained(output_directory)
        postprocessor.save_pretrained(output_directory)
    # Initialize optimizer and lr scheduler
    optimizer, lr_scheduler = build_optimizer_and_scheduler(policy, cfg, dataset_metadata.info["total_frames"], accelerator)

    # print only in main process
    accelerator.print("\n---policy_cfg", policy_cfg)
    accelerator.print(f"\n---Input features: {input_features}")
    accelerator.print(f"\n---Output features: {output_features}")
    accelerator.print(f"\n---camera_keys:", dataset_metadata.camera_keys)
    accelerator.print(f"\n---Original dataset features:", dataset_metadata.features) 

    num_total_params = sum(p.numel() for p in policy.parameters())
    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    accelerator.print(f"num_learnable_params={num_learnable_params}", f"num_total_params={num_total_params}")


    # Build dataset and dataloader
    delta_timestamps = build_delta_timestamps(dataset_metadata, policy_cfg)

    image_transforms = build_augmenter(cfg.training.RGB_Augmenter)
    dataset = LeRobotDataset(
        cfg.repoid,
        delta_timestamps=delta_timestamps,
        root=cfg.root,
        image_transforms=None,
    )
    accelerator.wait_for_everyone()
    # Training loop
    aug_step = insert_before_normalizer(preprocessor, AugmentationProcessorStep(image_transforms, dataset.meta.camera_keys))  # just for training
    
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    dataloader = DataLoader(
        dataset,
        num_workers=cfg.training.num_workers,
        batch_size=cfg.training.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=(device.type != "cpu"),
        drop_last=cfg.training.drop_last,
        prefetch_factor=2 if cfg.training.num_workers > 0 else None,
    )
    # Use accelerator to prepare data, model, and optimizer
    accelerator.wait_for_everyone()

    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, dataloader, lr_scheduler
    )


    # Initialize training state variables
    start_epoch = 0
    steps = 0
    best_loss = float('inf')

    # # ===== Resume logic (perfect resume for AMP & RNG) =====
    if cfg.training.resume and cfg.training.resume_timestamp:
        resume_path = Path(cfg.training.output_directory) / cfg.training.resume_timestamp
        accelerator.print("Resuming from:", resume_path)
        try:
            # Load state
            accelerator.load_state(resume_path / "epochlatest")
            if accelerator.is_main_process:
                latest_training_state = torch.load(resume_path / "training_latest_state.pth", map_location='cpu')
                steps = latest_training_state["steps"]
                start_epoch = latest_training_state["epoch"]
                best_loss = latest_training_state["best_loss"]
                accelerator.print(f"Resumed training from epoch {start_epoch}, step {steps}, best_loss {best_loss}")
        except Exception as e:
            accelerator.print("Failed to load checkpoint:", e, " Starting training from scratch.")
    else:
        accelerator.print("Training from scratch!")

    
    # Training loop
    for epoch in range(start_epoch, cfg.training.max_epoch):
        policy.train()

        # Use tqdm only on main process
        if accelerator.is_main_process:
            epoch_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.training.max_epoch}")
        else:
            epoch_bar = dataloader

        total_loss = 0.0
        batch_count = 0
        for batch in epoch_bar:
            batch = preprocessor(batch)
            with accelerator.accumulate(policy):
                # batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
                with accelerator.autocast():
                    loss, _ = policy.forward(batch)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(policy.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()
                    if accelerator.is_main_process:
                        if steps % cfg.training.log_freq == 0:
                            writer.add_scalar("train/loss", loss.item(), steps)
                            writer.add_scalar("train/lr", lr_scheduler.get_last_lr()[0], steps)
                            epoch_bar.set_postfix(loss=f"{loss.item():.3f}", step=steps, lr=lr_scheduler.get_last_lr()[0])
                    steps += 1
                    batch_count += 1
                    total_loss += accelerator.gather(loss).mean().item()

        total_loss = total_loss / batch_count if batch_count > 0 else total_loss
        
        # Log, save, and eval flags
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            # Update best loss
            if total_loss < best_loss:
                best_loss = total_loss
                unwrapped_policy = accelerator.unwrap_model(policy)
                unwrapped_policy.save_pretrained(output_directory / "epochbest")

            # Save checkpoint every N epochs
            if (epoch + 1) % cfg.training.save_freq_epoch == 0:
                unwrapped_policy = accelerator.unwrap_model(policy)
                unwrapped_policy.save_pretrained(output_directory / f"epoch{epoch+1}")

                # save latest epoch training state based on accelerator save_state
            accelerator.print("!!!!!!Saving latest epoch training state...,DON'T CTRL+C EXIT!!!!!!")
            accelerator.save_state(output_directory / "epochlatest")
            training_state = {
                "epoch": epoch+1, 
                "steps": steps,
                "best_loss": best_loss
            }
            torch.save(training_state, output_directory / "training_latest_state.pth")
            accelerator.print(f"Epoch {epoch+1} completed. Avg Loss: {total_loss:.4f}. Best Loss: {best_loss:.4f}")
        accelerator.wait_for_everyone()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        writer.close()
    
    accelerator.end_training()


if __name__ == "__main__":
    main()
