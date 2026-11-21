import numpy as np


def cosine_annealing(
    step: int, total_steps: int, lr_max: float, lr_min: float
) -> float:
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))
