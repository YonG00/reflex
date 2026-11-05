"""Wall clock phase boundaries; CUDA work is completed before each timestamp."""

import time
import torch


def timestamp(device):
    if str(device).startswith("cuda") and torch.cuda.is_initialized():
        torch.cuda.synchronize(device)
    return time.monotonic()
