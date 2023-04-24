import torch
import torch.nn.functional as F


def normalize(seq_data: torch.Tensor):
    """
    seq_data: [bs, seq_len, input size (5)]
    """
    assert seq_data.dim() == 3
    ret = F.normalize(seq_data, p=2, dim=1)
    return ret