from common.config import *
from common.args import get_args
from models.Forecaster import Forecaster
from models.Classifier import Classifier


def main():
    args = get_args()

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
    
    classifier = Classifier()
    classifier.load_from_pkl(classifier_save_path)
    

if __name__ == '__main__':
    main()
