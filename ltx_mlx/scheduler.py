"""FlowMatch Euler Discrete Scheduler — pure numpy, no diffusers.

Implements the exact sigma computation from diffusers LTXPipeline.

The pipeline computes sigmas as:
    1. linspace(1.0, 1/num_steps, num_steps) = [1.0, 0.75, 0.5, 0.25]
    2. calculate_shift(seq_len) to get mu
    3. Apply time_shift with mu to get shifted sigmas
    4. Append 0.0

Config (from models/LTX/scheduler/scheduler_config.json):
    base_shift: 0.95
    max_shift: 2.05
    base_image_seq_len: 1024
    num_train_timesteps: 1000
"""

import math
import numpy as np


def calculate_shift(
    seq_len: int,
    base_seq_len: int = 1024,
    max_seq_len: int = 4096,
    base_shift: float = 0.95,
    max_shift: float = 2.05,
) -> float:
    """Compute dynamic shift mu. Matches diffusers calculate_shift."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = seq_len * m + b
    return mu


def _time_shift(mu: float, sigma: np.ndarray) -> np.ndarray:
    """Apply exponential time shift: exp(mu) * s / (1 + (exp(mu)-1) * s)."""
    return math.exp(mu) * sigma / (1.0 + (math.exp(mu) - 1.0) * sigma)


def get_sigmas(num_steps: int, seq_len: int,
               base_seq_len: int = 1024,
               max_seq_len: int = 4096,
               base_shift: float = 0.95,
               max_shift: float = 2.05) -> np.ndarray:
    """Compute sigma schedule matching the diffusers LTX pipeline.

    Returns (num_steps + 1,) array: [sigma_0, sigma_1, ..., 0.0]
    """
    mu = calculate_shift(seq_len, base_seq_len, max_seq_len, base_shift, max_shift)

    # Linearly spaced input sigmas: [1.0, ..., 1/num_steps]
    sigmas_input = np.linspace(1.0, 1.0 / num_steps, num_steps)

    # Apply dynamic shift
    sigmas = _time_shift(mu, sigmas_input)

    # Append terminal 0
    sigmas = np.append(sigmas, 0.0)

    return sigmas.astype(np.float32)


def euler_step(model_output, sigma: float, sigma_next: float, sample):
    """Single Euler step for flow matching denoising."""
    return sample + (sigma_next - sigma) * model_output
