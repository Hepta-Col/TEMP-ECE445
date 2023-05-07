import argparse
import json
from common.config import *


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_forecaster', action='store_true')
    parser.add_argument('--train_classifier', action='store_true')
    parser.add_argument('--debug', action='store_true')
    
    #! crucial
    parser.add_argument('--granularity', type=str, choices=['hour', 'day'])
    parser.add_argument('--historical_length', type=int, default=48)
    parser.add_argument('--prediction_length', type=int, default=4)

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
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--eval_interval', type=int, default=10)
    parser.add_argument('--temperature_penalty', type=float, default=4)
    
    #! classifier
    parser.add_argument('--visualize_tree', action='store_true')

    args = parser.parse_args()
    
    return args


args = get_args()

if args.granularity == 'hour':
    args.batch_size = 512
    args.lr = 1e-2
    # args.num_epochs = 1550    # for penalty = 2
    args.num_epochs = 3000      # for penalty = 4
elif args.granularity == 'day':
    args.batch_size = 64
    args.lr = 1e-4
    # args.num_epochs = 15000   # for penalty = 2
    args.num_epochs = 30000     # for penalty = 4


#! input
data_root = "../data"
assert os.path.exists(data_root)
args.csv_path = os.path.join(data_root, "haining_weather.csv")
args.database_path = os.path.join(data_root, "weatherdata.db")

#! intermediate
save_root = "../saved_models/"
if not os.path.exists(save_root):
    os.makedirs(save_root)
args.forecaster_save_path = os.path.join(save_root, f"forecaster_{args.granularity}.pt")
args.classifier_save_path = os.path.join(save_root, "classifier.pkl")

#! output
f = lambda b: "[del] " if b else ""
args.out_root = os.path.join("../out", f(args.debug) + time)
args.figs_dir = os.path.join(args.out_root, "figs")
args.logs_dir = os.path.join(args.out_root, "logs")
os.makedirs(args.figs_dir)
os.makedirs(args.logs_dir)

args_path = os.path.join(args.out_root, "args.json")
json.dump(args.__dict__, open(args_path, "w"))
