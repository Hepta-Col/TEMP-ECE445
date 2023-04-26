import os
from common.args import args
from common.config import *
from utils.Trainers import ForecasterTrainer_V1, ClassifierTrainer_V1


def main():
    if args.train_forecaster:
        print("===================== Training Weather Forecaster =====================")
        forecaster_trainer = ForecasterTrainer_V1(args)
        forecaster_trainer.train()
    
    if args.train_classifier:
        print("===================== Training Weather Classifier =====================")
        classifier_trainer = ClassifierTrainer_V1(args)
        classifier_trainer.train()


if __name__ == '__main__':
    main()
    print("train.py: DONE!")
    open(os.path.join(args.out_root, "train-DONE"), "w").close()
