import argparse
from Trainer import ForecasterTrainer_V1


def _get_args():
    parser = argparse.ArgumentParser()

    #! crucial
    parser.add_argument('--sequence_length', type=int, default=2)   # 24

    #! dataset
    parser.add_argument('--csv_path', type=str, default="../data/haining_weather.csv")
    parser.add_argument('--train_test_ratio', type=float, default=9)

    #! training
    parser.add_argument('--num_epochs', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=4)    # 1024
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--eval_interval', type=int, default=10)

    #! forecaster
    parser.add_argument('--lstm_hidden_size', type=int, default=16)
    parser.add_argument('--lstm_num_layers', type=int, default=2)
    parser.add_argument('--lstm_dropout', type=float, default=0.2)
    
    parser.add_argument('--mlp_hidden_size', type=int, default=128)
    parser.add_argument('--mlp_num_layers', type=int, default=1)
    parser.add_argument('--mlp_dropout', type=float, default=0.2)

    args = parser.parse_args()
    
    args.device = 'cuda'
    
    return args


def main():
    args = _get_args()
    trainer = ForecasterTrainer_V1(args)
    trainer.train()


if __name__ == '__main__':
    main()
