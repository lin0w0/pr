"""Deterministic seeding helpers."""
import os
import random

import numpy as np
import torch


def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def set_deterministic(enabled=True):
    """Best-effort determinism. Some CUDA ops still have no deterministic kernel."""
    if enabled:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # TF32 (10-bit mantissa) is enabled for cuDNN convolutions by default.
        # Different batch sizes then select different reduced-precision kernels;
        # CCNet gradients were observed to differ by ~5.8% between batch 1 and
        # 16. Full FP32 keeps the SGM results batch-size stable to ~1e-3.
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            torch.set_float32_matmul_precision("highest")
        except AttributeError:  # older torch
            pass
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:  # older torch
            torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
