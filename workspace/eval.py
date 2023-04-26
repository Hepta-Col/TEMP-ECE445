import pdb
import os
import torch
from common.args import args
from common.config import *
from common.funcs import *
from utils.System import System
from utils.WeatherDataset import get_system_evaluation_dataloader


def main():
    system = System(args)
    
    dataloader = get_system_evaluation_dataloader(csv_path, args.historical_length, args.prediction_length)
    with torch.no_grad():
        for batch_id, (seq_batch, tgt_batch, descrp_batch) in enumerate(dataloader):
            history = seq_batch.squeeze()
            gt_data = tgt_batch.squeeze()
            gt_descriptions = [weather_descriptions[i.item()] for i in descrp_batch.squeeze()]
            predictions = system.predict_multi_step(history, args.prediction_length)
            
            gt_t_list = gt_data[:,0].squeeze().tolist()
            gt_p_list = gt_data[:,1].squeeze().tolist()
            gt_h_list = gt_data[:,2].squeeze().tolist()
            gt_w_list = gt_data[:,3].squeeze().tolist()
            gt = [gt_t_list, gt_p_list, gt_h_list, gt_w_list]
            
            pred_t_list = [pred.temperature for pred in predictions]
            pred_p_list = [pred.pressure for pred in predictions]
            pred_h_list = [pred.humidity for pred in predictions]
            pred_w_list = [pred.wind_speed for pred in predictions]
            pred = [pred_t_list, pred_p_list, pred_h_list, pred_w_list]
            
            x = list(range(args.prediction_length))
            names = ["temperature", "pressure", "humidity", "wind_speed"]
            for i in range(4):
                plt.figure()
                plt.plot(x, gt[i], color='r')
                plt.plot(x, pred[i], color='g')
                plt.savefig(os.path.join(args.figs_dir, f"{names[i]}.jpg"))
            
            break
            

if __name__ == '__main__':
    main()
    print("eval.py: DONE!")
    open(os.path.join(args.out_root, "eval-DONE"), "w").close()