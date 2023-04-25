import pdb
import os
import torch
from common.args import get_args
from common.config import *
from utils.System import System

import warnings
warnings.filterwarnings('ignore')


def main():
    args = get_args()
    system = System(args)
    
    data = torch.ones((24, 5))
    predictions = system.predict_multi_step(data, 24)
    

if __name__ == '__main__':
    main()
    print("inference.py: DONE!")
    open(os.path.join(out_root, "inference-DONE"), "w").close()
