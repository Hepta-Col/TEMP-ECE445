import pdb
import torch
from common.args import get_args
from common.config import *
from common.funcs import *
from utils.System import System
from utils.WeatherDataset import get_system_evaluation_dataloader


def main():
    args = get_args()
    system = System(args)
    
    dataloader = get_system_evaluation_dataloader(csv_path, args.historical_length, args.prediction_length)
    with torch.no_grad():
        for batch_id, (seq_batch, tgt_batch, descrp_batch) in enumerate(dataloader):
            history = seq_batch.squeeze()
            gt_data = tgt_batch.squeeze()
            gt_descriptions = [weather_descriptions[i.item()] for i in descrp_batch.squeeze()]
            data_list, description_list = system.predict_multi_step(history, args.prediction_length)
    

if __name__ == '__main__':
    main()
    print("eval.py: DONE!")
