import argparse
import json
from common.config import *


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_forecaster', action='store_true')
    parser.add_argument('--train_classifier', action='store_true')
    
    #! crucial
    parser.add_argument('--historical_length', type=int, default=24)
    parser.add_argument('--prediction_length', type=int, default=12)

    #! training
    parser.add_argument('--save_models', action='store_true')

    #! dataset
    parser.add_argument('--train_test_ratio', type=float, default=9)

    #! forecaster config
    parser.add_argument('--lstm_hidden_size', type=int, default=32)
    parser.add_argument('--lstm_num_layers', type=int, default=2)
    parser.add_argument('--lstm_dropout', type=float, default=0.)   #! don't change this
    
    parser.add_argument('--mlp_hidden_size', type=int, default=128)
    parser.add_argument('--mlp_num_layers', type=int, default=1)
    parser.add_argument('--mlp_dropout', type=float, default=0.)   #! don't change this

    #! forecaster training
    parser.add_argument('--num_epochs', type=int, default=3000)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--eval_interval', type=int, default=10)
    
    #! classifier
    parser.add_argument('--visualize_tree', action='store_true')

    """good combinations
    
    batch_size: 512

    lstm_hidden_size: 32
    lstm_num_layers: 2
    
    mlp_hidden_size: 128
    mlp_num_layers: 1
    """

    args = parser.parse_args()
    
    json.dump(args.__dict__, open(args_path, "w"))
    
    return args
