import random
import numpy as np
import torch

def save_rng_state(filepath: str) -> None:
    """Save RNG state for reproducibility."""
    state = {
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_rng"] = torch.cuda.get_rng_state_all()
    torch.save(state, filepath)


def load_rng_state(filepath: str) -> None:
    """Load RNG state for reproducibility."""
    state = torch.load(filepath, map_location="cpu", weights_only=False)
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    if torch.cuda.is_available() and "torch_cuda_rng" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda_rng"])


def worker_init_fn(worker_id: int) -> None:
    """Ensure each worker has a distinct reproducible seed."""
    # 让每个 worker 基于全局种子 + worker_id 派生
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)
