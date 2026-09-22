# ==============================
# ./src/utils/reproducibility.py
# ==============================

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 1989) -> None:

    """
    Sets global deterministic seeds across Python, NumPy, PyTorch, and CUDA/cuDNN.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:

    """
    Initializes PyTorch DataLoader worker processes with deterministic seeds.
    """
    
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

