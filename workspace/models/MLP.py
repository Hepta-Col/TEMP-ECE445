import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, 
                 num_layers, 
                 input_size,
                 hidden_size, 
                 output_size, 
                 dropout) -> None:
        super().__init__()

        layers = [nn.Linear(input_size, hidden_size)]

        assert num_layers > 0
        for i in range(num_layers):
            # Last layer (n_hidden x n_outputs)
            if i == num_layers - 1:
                layers.append(nn.ReLU())
                layers.append(nn.Linear(hidden_size, output_size))
            # All other layers (n_hidden x n_hidden) with dropout option
            else:
                layers.append(nn.ReLU())
                layers.append(nn.Linear(hidden_size, hidden_size))

        self.model = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.model(x)
        x = self.dropout(x)
        return x