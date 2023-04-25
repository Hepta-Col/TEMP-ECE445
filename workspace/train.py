from common.args import get_args
from utils.Trainers import ForecasterTrainer_V1, ClassifierTrainer_V1

import warnings
warnings.filterwarnings('ignore')


def main():
    args = get_args()
    print("===================== Training Weather Forecaster =====================")
    forecaster_trainer = ForecasterTrainer_V1(args)
    forecaster_trainer.train()
    
    print("===================== Training Weather Classifier =====================")
    classifier_trainer = ClassifierTrainer_V1(args)
    classifier_trainer.train()


if __name__ == '__main__':
    main()
    print("train.py: DONE!")
