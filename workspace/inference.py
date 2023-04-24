import pdb
import torch
from common.args import get_args
from utils.System import System


def main():
    args = get_args()
    system = System(args)
    
    test_data = torch.ones((24, 5))
    data_list, description_list = system.predict_multi_step(test_data, 5)
    
    pdb.set_trace()
    

if __name__ == '__main__':
    main()
    print("main.py: DONE!")