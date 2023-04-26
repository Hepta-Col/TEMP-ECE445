import pdb
import os
import torch
import pandas as pd
import sqlite3 as sql
from common.args import args
from common.config import *
from utils.System import System


def main():
    # system = System(args)
    
    # data = torch.ones((24, 5))
    # predictions = system.predict_multi_step(data, 2)
    
    num_lines = 0
    
    pdb.set_trace()
    
    while True:
        with sql.connect(database_path) as con:
            df = pd.read_sql("SELECT * FROM weatherdata", con=con)
            
            if df.shape[0] == num_lines:
                continue
            
            num_lines = df.shape[0]

            print(df.shape)
            print(df.dtypes)
            print(df.head())        


if __name__ == '__main__':
    main()
    print("run.py: DONE!")
    open(os.path.join(args.out_root, "run-DONE"), "w").close()
