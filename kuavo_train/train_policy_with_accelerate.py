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
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.utils.random_utils import set_seed

from kuavo_train.wrapper.policy.diffusion.DiffusionPolicyWrapper import CustomDiffusionPolicyWrapper
from kuavo_train.wrapper.dataset.LeRobotDatasetWrapper import CustomLeRobotDataset
from kuavo_train.utils.augmenter import crop_image, resize_image, DeterministicAugmenterColor
from kuavo_train.utils.utils import save_rng_state, load_rng_state
from lerobot.policies.act.modeling_act import ACTPolicy
from diffusers.optimization import get_scheduler
from utils.transforms import ImageTransforms, ImageTransformsConfig, ImageTransformConfig

from functools import partial
from contextlib import nullcontext


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

def build_policy(name, policy_cfg, dataset_stats):
    policy = {
        "diffusion": CustomDiffusionPolicyWrapper,
        "act": ACTPolicy,
    }[name](policy_cfg, dataset_stats)
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
    policy = build_policy(cfg.policy_name, policy_cfg, dataset_stats=dataset_metadata.stats)

    # Initialize optimizer and lr scheduler
    optimizer, lr_scheduler = build_optimizer_and_scheduler(policy, cfg, dataset_metadata.info["total_frames"], accelerator)

    # print only in main process
    if accelerator.is_main_process:
        print("policy_cfg", policy_cfg)
        print(f"Input features: {input_features}")
        print(f"Output features: {output_features}")
        print("camera_keys:", dataset_metadata.camera_keys)
        print("Original dataset features:", dataset_metadata.features) 
        num_total_params = sum(p.numel() for p in policy.parameters())
        num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        print(f"{num_learnable_params=} ({num_learnable_params})", f"{num_total_params=} ({num_total_params})")
    accelerator.wait_for_everyone()

    # Initialize training state variables
    start_epoch = 0
    steps = 0
    best_loss = float('inf')

    # # ===== Resume logic (perfect resume for AMP & RNG) =====
    if cfg.training.resume and cfg.training.resume_timestamp:
        resume_path = Path(cfg.training.output_directory) / cfg.training.resume_timestamp
        print("Resuming from:", resume_path)
        try:
            # Load state
            accelerator.load_state(resume_path / "latest_epoch")
            if accelerator.is_main_process:
                best_training_state = torch.load(resume_path / "training_latest_state.pth", map_location='cpu')
                steps = best_training_state["steps"]
                start_epoch = best_training_state["epoch"]
                best_loss = best_training_state["best_loss"]
                print(f"Resumed training from epoch {start_epoch}, step {steps}, best_loss {best_loss}") 
        except Exception as e:
            print("Failed to load checkpoint:", e)
            return
    else:
        print("Training from scratch!")


    # Build dataset and dataloader
    delta_timestamps = build_delta_timestamps(dataset_metadata, policy_cfg)

    image_transforms = build_augmenter(cfg.training.RGB_Augmenter)                 # TODO  add angle tranforms
    dataset = LeRobotDataset(
        cfg.repoid,
        delta_timestamps=delta_timestamps,
        root=cfg.root,
        image_transforms=image_transforms,
    )


    dataloader = DataLoader(
        dataset,
        num_workers=cfg.training.num_workers,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        pin_memory=(device.type != "cpu"),
        drop_last=cfg.training.drop_last,
        prefetch_factor=1,
    )

    # Use accelerator to prepare data, model, and optimizer
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, dataloader, lr_scheduler
    )

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
            with accelerator.accumulate(policy):
                batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

                loss, _ = policy.forward(batch)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(policy.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()

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
                unwrapped_policy.save_pretrained(output_directory / "best")

                writer.add_scalar("train/epoch: ", epoch, steps)
                writer.add_scalar("train/loss: ", total_loss, steps)
                writer.add_scalar("train/lr: ", lr_scheduler.get_last_lr()[0], steps)

            # Save checkpoint every N epochs
            if (epoch + 1) % cfg.training.save_freq_epoch == 0:
                unwrapped_policy = accelerator.unwrap_model(policy)
                unwrapped_policy.save_pretrained(output_directory / f"epoch{epoch+1}")

                # save latest epoch training state based on accelerator save_state
                accelerator.save_state(output_directory / "latest_epoch", safe_serialization=False)
                training_state = {
                    "epoch": epoch+1, 
                    "steps": steps,
                    "best_loss": best_loss
                }
                torch.save(training_state, output_directory / "training_latest_state.pth")


    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        writer.close()
    
    accelerator.end_training()


if __name__ == "__main__":
    main()
