import argparse

from Trainer import Trainer


def _get_args():
    parser = argparse.ArgumentParser()

    #! paths
    parser.add_argument('--csv_path', type=str, default="./data/haining_weather.csv")

    #! hyperparameters
    parser.add_argument('--look_back_window', type=int, default=48, help="how many previous hours are considered for prediction?")
    parser.add_argument('--predict_window', type=int, default=24, help="how many future hours are going to be predicted?")

    parser.add_argument('--batch_size', type=int, default=32)
    
    parser.add_argument('--train_test_ratio', type=float, default=9)

    #! LSTM

    

    args = parser.parse_args()
    return args


def main():
    args = _get_args()
    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    main()