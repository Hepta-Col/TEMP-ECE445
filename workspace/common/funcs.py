import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def normalize(seq_data: torch.Tensor):
    """
    seq_data: [bs, seq_len, input size (5)]
    """
    assert seq_data.dim() == 3
    ret = F.normalize(seq_data, p=2, dim=1)
    return ret

def plot_single_curve(x, y, plot_path):
    plt.figure()
    plt.plot(x, y)
    plt.savefig(plot_path)
    
def avg(arr):
    assert len(arr) > 0
    return sum(arr) / len(arr)

