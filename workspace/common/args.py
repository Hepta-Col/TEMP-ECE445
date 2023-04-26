import argparse
import json
from common.config import *


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_forecaster', action='store_true')
    parser.add_argument('--train_classifier', action='store_true')
    parser.add_argument('--debug', action='store_true')
    
    #! crucial
    parser.add_argument('--historical_length', type=int, default=48)
    parser.add_argument('--prediction_length', type=int, default=12)

    #! saving
    parser.add_argument('--save_models', action='store_true')

    #! dataset
    parser.add_argument('--train_test_ratio', type=float, default=9)

    #! forecaster config
    parser.add_argument('--lstm_hidden_size', type=int, default=64)
    parser.add_argument('--lstm_num_layers', type=int, default=2)
    parser.add_argument('--lstm_dropout', type=float, default=0.)   #! don't change this
    
    parser.add_argument('--mlp_hidden_size', type=int, default=128)
    parser.add_argument('--mlp_num_layers', type=int, default=2)
    parser.add_argument('--mlp_dropout', type=float, default=0.)   #! don't change this

    #! forecaster training
    parser.add_argument('--num_epochs', type=int, default=1990)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--eval_interval', type=int, default=10)
    
    #! classifier
    parser.add_argument('--visualize_tree', action='store_true')

    args = parser.parse_args()
    
    return args


args = get_args()

#! output
f = lambda b: "[del] " if b else ""
args.out_root = os.path.join("../out", f(args.debug) + time)
args.figs_dir = os.path.join(args.out_root, "figs")
args.logs_dir = os.path.join(args.out_root, "logs")
os.makedirs(args.figs_dir)
os.makedirs(args.logs_dir)
args_path = os.path.join(args.out_root, "args.json")
json.dump(args.__dict__, open(args_path, "w"))
