import torch
import torch.nn as nn
from models.Forecaster import Forecaster
from models.Classifier import Classifier
from common.config import *


class System(object):
    def __init__(self, args):
        super().__init__()

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
        forecaster = Forecaster(lstm_config, mlp_config).to(device)
        forecaster.load_state_dict(torch.load(forecaster_save_path))

        """forecaster in & out:
        y = forecaster(x)
        x: torch.Tensor([batch size, sequence length, input size (5: [T, P, H, W, M])])
        y: torch.Tensor([batch size, sequence length, output size (4: [T, P, H, W])])
        """

        print("==> Building classifier...")
        classifier = Classifier()
        classifier.load_from_pkl(classifier_save_path)
    
        """classifier in & out:
        y = classifier.predict(X)
        X: np.array([..., 5 (T, P, H, W, M)])
        y: List[str]
        """
        
    
        
        