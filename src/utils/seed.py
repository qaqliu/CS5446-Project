"""Initialize training RNGs and isolate validation draws from training."""

import random
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import torch


def set_random_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch; later draws advance normally."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def preserve_rng() -> Iterator[None]:
    """Restore Python, NumPy and PyTorch RNGs after read-only validation."""
    state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(state[0])
        np.random.set_state(state[1])
        torch.set_rng_state(state[2])
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
