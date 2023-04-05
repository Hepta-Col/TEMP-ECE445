import argparse
from utils.Trainers import ForecasterTrainer_V1, ClassifierTrainer_V1


def _get_args():
    parser = argparse.ArgumentParser()

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
    parser.add_argument('--num_epochs', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--eval_interval', type=int, default=10)

    """good combinations
    
    batch_size: 512

    lstm_hidden_size: 32
    lstm_num_layers: 2
    
    mlp_hidden_size: 128
    mlp_num_layers: 1
    """

    args = parser.parse_args()
    
    return args


def main():
    args = _get_args()
    print("===================== Training Weather Forecaster =====================")
    forecaster_trainer = ForecasterTrainer_V1(args)
    forecaster_trainer.train()
    
    print("===================== Training Weather Classifier =====================")
    classifier_trainer = ClassifierTrainer_V1(args)
    classifier_trainer.train()
    
    print("ALL DONE!")


if __name__ == '__main__':
    main()
