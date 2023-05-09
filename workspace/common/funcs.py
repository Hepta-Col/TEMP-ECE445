import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def normalize(seq_data: torch.Tensor):
    """
    seq_data: [bs, seq_len, input size]
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

def error(pred, gt, type):
    if type == 'rel':
        return round(float(abs(gt-pred)/gt*100), 2)
    elif type == 'abs':
        return round(float(abs(pred - gt)), 2)

def mid(x1, x2, ratio):
    small = min(x1, x2)
    large = max(x1, x2)
    return small + (large - small) * ratio

def remove_anomalies(arr):
    assert len(arr) > 2
    sorted_arr = sorted(arr)
    return sorted_arr[1:-1]
