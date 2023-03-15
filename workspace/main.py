import argparse

from Trainer import ForecasterTrainer


def _get_args():
    parser = argparse.ArgumentParser()

    #! paths
    parser.add_argument('--csv_path', type=str, default="./data/haining_weather.csv")

    #! hyperparameters
    # basics
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--train_test_ratio', type=float, default=9)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--eval_interval', type=int, default=10)

    # auto-regression dataset
    parser.add_argument('--look_back_window', type=int, default=48, help="how many previous hours are considered for prediction?")
    parser.add_argument('--predict_window', type=int, default=24, help="how many future hours are going to be predicted?")

    # forecaster
    parser.add_argument('--input_size', type=int, default=1)
    parser.add_argument('--hidden_size', type=int, default=32)
    parser.add_argument('--num_lstm_layers', type=int, default=1)
    parser.add_argument('--num_deep_layers', type=int, default=4)
    parser.add_argument('--last_drop_rate', type=float, default=0.2)
    parser.add_argument('--mid_drop_rate', type=float, default=0.2)

    args = parser.parse_args()
    return args


def main():
    args = _get_args()
    forecaster_trainer = ForecasterTrainer(args)
    forecaster_trainer.train()


if __name__ == '__main__':
    main()