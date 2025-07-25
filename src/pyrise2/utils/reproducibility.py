import numpy as np

def set_seed(seed):
    """Set seed for reproducibility."""
    np.random.seed(seed)
    print(f"Seed set to {seed}.")
