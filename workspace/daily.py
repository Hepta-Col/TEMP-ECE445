import pdb
import os
import torch
import pandas as pd
import time as tm
from common.args import args
from common.config import *
from common.funcs import *
from utils.System import System


def main():
    print("==> Creating forecasting system...")
    system = System(args)
    
    test_hour_data = pd.read_csv("")
    test_day_data = pd.read_csv("")
    assert test_hour_data.shape == (48, 5) and test_day_data.shape == (48, 5)
    test_hour_data = torch.tensor(test_hour_data.values)
    test_day_data = torch.tensor(test_day_data.values)
    
    month_col = torch.tensor([float(datetime.now().month) for _ in range(48)]).unsqueeze(1)
    model_input = torch.cat((test_day_data, month_col), dim=1)
    assert model_input.shape == (48, 6)
    predictions = system.predict_multi_step(model_input, args.prediction_length)
    
    for p in predictions:
        print(p)


if __name__ == '__main__':
    main()
    print("test.py: DONE!")
    open(os.path.join(args.out_root, "run-DONE"), "w").close()
