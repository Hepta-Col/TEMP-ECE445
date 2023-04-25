import pdb
import torch
from common.args import get_args
from utils.System import System

import warnings
warnings.filterwarnings('ignore')


def main():
    args = get_args()
    system = System(args)
    
    data = torch.ones((24, 5))
    prediction = system.predict_multi_step(data, 2)
    print(prediction)
    

if __name__ == '__main__':
    main()
    print("inference.py: DONE!")
