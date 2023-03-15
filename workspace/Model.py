import torch
import torch.nn as nn


class Forecaster(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        self.args = args

        self.lstm = nn.LSTM(input_size=args.input_size,
                            hidden_size=args.hidden_size,
                            num_layers=args.num_lstm_layers,
                            batch_first=True)

        # first dense after lstm
        self.fc = nn.Linear(args.hidden_size * args.look_back_window, args.hidden_size)

        self.last_dropout = nn.Dropout(args.last_drop_rate)

        # Create fully connected layers (hidden_size x num_deep_layers)
        dnn_layers = []
        for i in range(args.num_deep_layers):
            # Last layer (n_hidden x n_outputs)
            if i == args.num_deep_layers - 1:
                dnn_layers.append(nn.ReLU())
                dnn_layers.append(nn.Linear(args.hidden_size, args.predict_window))
            # All other layers (n_hidden x n_hidden) with dropout option
            else:
                dnn_layers.append(nn.ReLU())
                dnn_layers.append(nn.Linear(args.hidden_size, args.hidden_size))
                dnn_layers.append(nn.Dropout(args.mid_drop_rate))
        self.dnn = nn.Sequential(*dnn_layers)

    def forward(self, x):
        """
        x: [batch size, sequence length, input size (1)]
        """
        # Forward Pass
        x, _ = self.lstm(x) # x: [batch size, sequence length, hidden size]
        x = x.contiguous().view(x.shape[0], -1) # x: [batch size, sequence length * hidden size]
        x = self.last_dropout(x)
        x = self.fc(x) # x: [batch size, hidden size]
        output = self.dnn(x) # x: [batch size, prediction window size]
        """
        output: [batch size, prediction window size]
        """
        return output


class Classifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        pass