import torch.nn as nn
from models.MLP import MLP


class Forecaster(nn.Module):
    def __init__(self, lstm_config, mlp_config) -> None:
        super().__init__()

        print("==> Creating model...")
        self.lstm = nn.LSTM(**lstm_config)
        self.mlp = MLP(**mlp_config)

    def forward(self, x):
        """
        x: [batch size, sequence length, input size (6, [T, P, H, W, R, M])]
        output: [batch size, sequence length, output size (5, [T, P, H, W, R])]
        """
        x, _ = self.lstm(x)
        batch_size, sequence_length, hidden_size = x.shape
        x = x.contiguous().view(-1, hidden_size)
        x = self.mlp(x)
        _, output_size = x.shape
        x = x.contiguous().view(batch_size, sequence_length, output_size)
        
        return x
