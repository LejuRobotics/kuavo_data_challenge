import os
import builtins
from collections import deque
from pathlib import Path
from typing import TypeVar
from typing_extensions import Unpack

import torch
from torch import Tensor

from lerobot.policies.pretrained import PreTrainedPolicy, ActionSelectKwargs
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from kuavo_train.wrapper.policy.gr00t_n1d5.Gr00tN1d5ConfigWrapper import CustomGr00tN1d5ConfigWrapper
from kuavo_train.wrapper.policy.gr00t_n1d5.Gr00tN1d5ModelWrapper import CustomGr00tN1d5ModelWrapper
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from huggingface_hub.errors import HfHubHTTPError


T = TypeVar("T", bound="CustomGr00tN1d5PolicyWrapper")


class CustomGr00tN1d5PolicyWrapper(PreTrainedPolicy):
    """
    Custom wrapper around GR00T N1.5 model for LeRobot integration.
    
    This wrapper uses CustomGr00tN1d5ConfigWrapper and CustomGr00tN1d5ModelWrapper
    to provide a consistent interface with the training framework.
    """
    
    name = "gr00t_n1d5"
    config_class = CustomGr00tN1d5ConfigWrapper
    
    def __init__(self, config: CustomGr00tN1d5ConfigWrapper, rtc_processor: RTCProcessor | None = None):
        """Initialize GR00T policy wrapper."""
        super().__init__(config)
        config.validate_features()
        self.config = config
        
        self.init_rtc_processor()
        
        # Initialize GR00T model using our custom model wrapper
        self._groot_model = CustomGr00tN1d5ModelWrapper(config, rtc_processor=self.rtc_processor)
        
        self.reset()
    
    def init_rtc_processor(self):
        """Initialize RTC processor if RTC is enabled in config."""
        self.rtc_processor = None
        
        # Create processor if config provided
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)
            
            model_value = getattr(self, "_groot_model", None)
            if model_value is not None:
                model_value.rtc_processor = self.rtc_processor
    
    def reset(self):
        """Reset policy state when environment resets."""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)
    
    def get_optim_params(self) -> list[dict]:
        """
        Return optimizer parameters.
        
        Returns:
            List of parameter groups for optimizer
        """
        return list(self.parameters())
    
    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """
        Training forward pass.
        
        Delegates to GR00T model.forward when inputs are compatible.
        
        Args:
            batch: Input batch dictionary
            
        Returns:
            Tuple of (loss, loss_dict)
        """
        # Build a clean input dict for GR00T: keep only tensors GR00T consumes
        allowed_base = {"state", "state_mask", "action", "action_mask", "embodiment_id"}
        groot_inputs = {
            k: v
            for k, v in batch.items()
            if (k in allowed_base or k.startswith("eagle_")) and not (k.startswith("next.") or k == "info")
        }
        # Get device from model parameters
        device = next(self.parameters()).device
        
        # Run GR00T forward under bf16 autocast when enabled to reduce activation memory
        # Rationale: Matches original GR00T finetuning (bf16 compute, fp32 params) and avoids fp32 upcasts.
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=self.config.use_bf16):
            outputs = self._groot_model.forward(groot_inputs)
        
        # Isaac-GR00T returns a BatchFeature; loss key is typically 'loss'
        loss = outputs.get("loss")
        
        # Extract all metrics from outputs for logging (e.g., arm_loss, claw_loss, sigma_arm, etc.)
        loss_dict = {}
        if isinstance(outputs, dict) or hasattr(outputs, "data"):
            output_data = outputs.data if hasattr(outputs, "data") else outputs
            for key, value in output_data.items():
                if isinstance(value, torch.Tensor):
                    if value.numel() == 1:  # Scalar tensor
                        loss_dict[key] = value.item()
                    else:
                        # Skip non-scalar tensors
                        continue
                elif isinstance(value, (int, float)):
                    loss_dict[key] = value
                # Skip other types (e.g., None, complex types)
        
        # Ensure loss is always in loss_dict
        if "loss" not in loss_dict:
            loss_dict["loss"] = loss.item() if isinstance(loss, torch.Tensor) else loss
        
        return loss, loss_dict
    
    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        """
        Predict a chunk of actions for inference by delegating to GR00T.
        
        Returns a tensor of shape (B, n_action_steps, action_dim).
        
        Args:
            batch: Input batch dictionary
            **kwargs: Additional arguments for action selection
            
        Returns:
            Action predictions tensor
        """
        self.eval()
        
        # Build a clean input dict for GR00T: keep only tensors GR00T consumes
        # Preprocessing is handled by the processor pipeline, so we just filter the batch
        # NOTE: During inference, we should NOT pass action/action_mask (that's what we're predicting)
        allowed_base = {"state", "state_mask", "embodiment_id"}
        groot_inputs = {
            k: v
            for k, v in batch.items()
            if (k in allowed_base or k.startswith("eagle_")) and not (k.startswith("next.") or k == "info")
        }
        
        # Get device from model parameters
        device = next(self.parameters()).device
        
        # Use bf16 autocast for inference to keep memory low and match backbone dtype
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=self.config.use_bf16):
            outputs = self._groot_model.get_action(groot_inputs, **kwargs)
        
        actions = outputs.get("action_pred")
        
        original_action_dim = self.config.output_features["action"].shape[0]
        actions = actions[:, :, :original_action_dim]
        
        return actions
    
    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """
        Select single action from action queue.
        
        Args:
            batch: Input batch dictionary
            
        Returns:
            Single action tensor
        """
        assert not self._groot_model._rtc_enabled(), (
            "RTC is not supported for select_action, use it with predict_action_chunk"
        )
        self.eval()
        
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()
    
    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: CustomGr00tN1d5ConfigWrapper | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = False,
        **kwargs,
    ) -> T:
        """
        Load a pretrained policy from a directory or HuggingFace Hub.
        
        The policy is set in evaluation mode by default using `policy.eval()`
        (dropout modules are deactivated). To train it, you should first set
        it back in training mode with `policy.train()`.
        
        Args:
            pretrained_name_or_path: Path to model directory or HuggingFace Hub ID
            config: Optional config object (if None, will be loaded from model directory)
            force_download: Whether to force download from Hub
            resume_download: Whether to resume download
            proxies: Optional proxy configuration
            token: Optional authentication token
            cache_dir: Optional cache directory
            local_files_only: Whether to use only local files
            revision: Optional revision/branch name
            strict: Whether to strictly enforce that the keys match
            **kwargs: Additional arguments
            
        Returns:
            Loaded policy instance (in eval mode)
        """
        if config is None:
            config = CustomGr00tN1d5ConfigWrapper.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )
        
        model_id = str(pretrained_name_or_path)
        
        # Create instance
        instance = cls(config, **kwargs)
        
        # Load weights if available
        if os.path.isdir(model_id):
            print("Loading weights from local directory")
            model_file = os.path.join(model_id, SAFETENSORS_SINGLE_FILE)
            policy = cls._load_as_safetensor(instance, model_file, config.device, strict)
        else:
            try:
                model_file = hf_hub_download(
                    repo_id=model_id,
                    filename=SAFETENSORS_SINGLE_FILE,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
                policy = cls._load_as_safetensor(instance, model_file, config.device, strict)
            except HfHubHTTPError as e:
                raise FileNotFoundError(
                    f"{SAFETENSORS_SINGLE_FILE} not found on the HuggingFace Hub in {model_id}"
                ) from e
        
        policy.to(config.device)
        policy.eval()
        return policy
    
    # -------------------------
    # Internal helpers
    # -------------------------
    def _handle_flash_attention_compatibility(self) -> None:
        """Handle Flash Attention compatibility issues by setting environment variables.
        
        This addresses the common 'undefined symbol' error that occurs when Flash Attention
        is compiled against a different PyTorch version than what's currently installed.
        """
        # Set environment variables to handle Flash Attention compatibility
        os.environ.setdefault("FLASH_ATTENTION_FORCE_BUILD", "0")
        os.environ.setdefault("FLASH_ATTENTION_SKIP_CUDA_BUILD", "0")
        
        # Try to import flash_attn and handle failures gracefully
        try:
            import flash_attn
            print(f"[GROOT] Flash Attention version: {flash_attn.__version__}")
        except ImportError as e:
            print(f"[GROOT] Flash Attention not available: {e}")
            print("[GROOT] Will use fallback attention mechanism")
        except Exception as e:
            if "undefined symbol" in str(e):
                print(f"[GROOT] Flash Attention compatibility issue detected: {e}")
                print("[GROOT] This is likely due to PyTorch/Flash Attention version mismatch")
                print("[GROOT] Consider reinstalling Flash Attention with compatible version:")
                print("  pip uninstall flash-attn")
                print("  pip install --no-build-isolation flash-attn==2.6.3")
                print("[GROOT] Continuing with fallback attention mechanism")
            else:
                print(f"[GROOT] Flash Attention error: {e}")
                print("[GROOT] Continuing with fallback attention mechanism")
