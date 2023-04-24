import pdb
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from tqdm import trange
from models.Forecaster import Forecaster
from models.Classifier import Classifier
from common.config import *
from common.funcs import *


class System(object):
    def __init__(self, args):
        super().__init__()

        self.args = args

        print("==> Building forecaster...")
        lstm_config = {
            'input_size': len(names_for_input_features),
            'hidden_size': args.lstm_hidden_size,
            'num_layers': args.lstm_num_layers,
            'dropout': args.lstm_dropout,
            'batch_first': True,
        }
        mlp_config = {
            'num_layers': args.mlp_num_layers,
            'input_size': args.lstm_hidden_size,
            'hidden_size': args.mlp_hidden_size,
            'output_size': len(names_for_output_features),
            'dropout': args.mlp_dropout,
        }
        self.forecaster = Forecaster(lstm_config, mlp_config).to(device)
        self.forecaster.load_state_dict(torch.load(forecaster_save_path))
        self.forecaster.eval()

        """forecaster in & out:
        y = forecaster(x)
        x: torch.Tensor([batch size, sequence length, input size (5: [T, P, H, W, M])])
        y: torch.Tensor([batch size, sequence length, output size (4: [T, P, H, W])])
        """

        print("==> Building classifier...")
        self.classifier = Classifier()
        self.classifier.load_from_pkl(classifier_save_path)
    
        """classifier in & out:
        y = classifier.predict(X)
        X: np.array([..., 5 (T, P, H, W, M)])
        y: List[str]
        """
        
        self.month = datetime.now().month

    def predict_single_step(self, historical_data):
        """get the next hour data based on historical data

        Args:
            historical_data (torch.Tensor): a Tensor with shape (24, 5) 
        """
        assert historical_data.shape == (self.args.historical_length, len(names_for_input_features))     # (24, 5)
        historical_data = normalize(historical_data.unsqueeze(0)).to(device)
        with torch.no_grad():
            pred = self.forecaster(historical_data)     # (1, 24, 4)
        next_hour_data = pred.squeeze()[-1].cpu()       # (4)
        classifier_input = torch.cat((next_hour_data, torch.tensor(self.month).unsqueeze(0))).unsqueeze(0).numpy()
        next_hour_description = self.classifier.predict(classifier_input)[0]
        return next_hour_data, next_hour_description
    
    def predict_multi_step(self, historical_data, num_steps):
        """get a series of future data based on historical data

        Args:
            historical_data (torch.Tensor): a Tensor with shape (24, 5) 
            num_steps (int): number of future hours to predict
        """
        data_list = []
        description_list = []
        input_data = historical_data
        for _ in range(num_steps):
            next_hour_data, next_hour_description = self.predict_single_step(input_data)
            data_list.append(next_hour_data)
            description_list.append(next_hour_description)
            new_line = torch.cat((next_hour_data, torch.tensor(self.month).unsqueeze(0))).unsqueeze(0)
            input_data = torch.cat((input_data[1:], new_line), dim=0)
        
        return data_list, description_list
