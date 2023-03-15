import torch
import torch.nn as nn


class Forecaster(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        self.args = args

        self.lstm = nn.LSTM(input_size=args.input_size,
                            hidden_size=args.hidden_size,
                            num_layers=args.num_layers,
                            batch_first=True)

        # first dense after lstm
        self.fc = nn.Linear(args.hidden_size * args.look_back_window, args.hidden_size)

        self.last_dropout = nn.Dropout(p=args.last_drop_rate)

        # Create fully connected layers (hidden_size x num_deep_layers)
        dnn_layers = []
        for i in range(args.num_deep_layers):
            # Last layer (n_hidden x n_outputs)
            if i == args.num_deep_layers - 1:
                dnn_layers.append(nn.ReLU())
                dnn_layers.append(nn.Linear(args.hidden_size, args.prediction_window))
            # All other layers (n_hidden x n_hidden) with dropout option
            else:
                dnn_layers.append(nn.ReLU())
                dnn_layers.append(nn.Linear(args.hidden_size, args.hidden_size))
                dnn_layers.append(nn.Dropout(args.mid_drop_rate))
        # compile DNN layers
        self.dnn = nn.Sequential(*dnn_layers)

    def forward(self, x):
        # Initialize hidden state
        hidden_state = torch.zeros(self.args.num_deep_layerss, x.shape[0], self.args.hidden_size)
        cell_state = torch.zeros(self.args.num_deep_layers, x.shape[0], self.args.hidden_size)

        # Forward Pass
        x, h = self.lstm(x, (hidden_state, cell_state)) # LSTM
        x = self.dropout(x.contiguous().view(x.shape[0], -1)) # Flatten lstm out 
        x = self.fc(x) # First Dense
        return self.dnn(x) # Pass forward through fully connected DNN.
    