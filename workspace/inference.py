import pdb
import torch
from common.args import get_args
from utils.System import System


def main():
    args = get_args()
    system = System(args)
    
    test_data = torch.ones((24, 5))
    
    out = system.predict_single_step(test_data)
    
    

if __name__ == '__main__':
    main()
