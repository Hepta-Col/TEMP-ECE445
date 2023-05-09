import pdb
import os
import torch
import pandas as pd
import sqlite3 as sql
from common.args import args
from common.config import *
from utils.System import System

from demo.search import retrieve_record


def main():
    # system = System(args)
    
    # data = torch.ones((48, 6))  # (historical length, 6 features TTPHWM) 
    # predictions = system.predict_multi_step(data, 2)
    # for p in predictions:
    #     print(p)
    
    # num_lines = 0
    # while True:
    #     with sql.connect(args.database_path) as con:
    #         df = pd.read_sql("SELECT * FROM weatherdata", con=con)
            
    #         if df.shape[0] == num_lines:
    #             continue
            
    #         num_lines = df.shape[0]

    #         print(df.shape)
    #         print(df.dtypes)
    #         print(df.head())        
    
    record = retrieve_record()


if __name__ == '__main__':
    main()
    print("run.py: DONE!")
    open(os.path.join(args.out_root, "run-DONE"), "w").close()
