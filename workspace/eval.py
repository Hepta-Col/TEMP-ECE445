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
    
    dataloader = get_system_evaluation_dataloader(args.csv_path, args.historical_length, args.prediction_length, args.granularity)
    with torch.no_grad():
        for batch_id, (seq_batch, tgt_batch, descrp_batch) in enumerate(dataloader):
            if batch_id % (len(dataloader) // 10) != 0:
                continue
            
            history = seq_batch.squeeze()
            gt_data = tgt_batch.squeeze()
            gt_descriptions = [weather_descriptions[i.item()] for i in descrp_batch.squeeze()]
            predictions = system.predict_multi_step(history, args.prediction_length)
            
            gt_t_min_list = gt_data[:,0].squeeze().tolist()
            gt_t_max_list = gt_data[:,1].squeeze().tolist()
            gt_p_list = gt_data[:,2].squeeze().tolist()
            gt_h_list = gt_data[:,3].squeeze().tolist()
            gt_w_list = gt_data[:,4].squeeze().tolist()
            gt = [gt_t_min_list, gt_t_max_list, gt_p_list, gt_h_list, gt_w_list]
            
            pred_t_min_list = [pred.temp_min for pred in predictions]
            pred_t_max_list = [pred.temp_max for pred in predictions]
            pred_p_list = [pred.pressure for pred in predictions]
            pred_h_list = [pred.humidity for pred in predictions]
            pred_w_list = [pred.wind_speed for pred in predictions]
            pred = [pred_t_min_list, pred_t_max_list, pred_p_list, pred_h_list, pred_w_list]
            
            x = np.arange(args.prediction_length).astype(dtype=int)
            names = ["temp_min", "temp_max", "pressure", "humidity", "wind_speed"]
            units = ["Degree C", "Degree C", "hPa", "%", "m/s"]
            assert len(names) == len(units)
            fig, axs = plt.subplots(len(names), 1, sharex=True, figsize=(8,10))
            axs = axs.ravel()
            for i in range(len(names)):
                axs[i].plot(x, gt[i], color='r', label='ground truth', marker='o')
                axs[i].plot(x, pred[i], color='g', label='prediction', marker='o')
                assert len(x) == len(pred[i]) == len(gt[i])
                for j in range(len(x)):
                    axs[i].annotate(f"error={error(pred[i][j], gt[i][j])}", (x[j], (pred[i][j] + gt[i][j])/2))
                if i == 0:
                    axs[i].legend()
                if i == len(names) - 1:
                    axs[i].set_xlabel(f'{args.granularity}')
                axs[i].set_ylabel(f'{names[i]}, {units[i]}')
            fig.tight_layout()
            savefig_path = os.path.join(args.figs_dir, f"Batch {batch_id}-Pred V.S. GT ({args.granularity}).jpg")
            plt.savefig(savefig_path)


if __name__ == '__main__':
    main()
    print("eval.py: DONE!")
    open(os.path.join(args.out_root, "eval-DONE"), "w").close()