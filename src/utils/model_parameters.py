# ===============================
# ./src/utils/model_parameters.py
# ===============================

from typing import Dict
import torch.nn as nn


def count_parameters(model: nn.Module) -> Dict[str, int]:

    """
    Counts the total and trainable parameters of a PyTorch model.

    Args:
        model: PyTorch model to analyze.

    Returns:
        Dictionary containing total and trainable parameter counts.
    """

    total_params = sum(parameter.numel() for parameter in model.parameters())

    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params
    }

