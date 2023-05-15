import pdb
import os
import torch
import pandas as pd
from common.args import args
from common.config import *
from common.funcs import *
from utils.System import System


def main():
    print("==> Creating forecasting system...")
    print(args.granularity)
    system = System(args)
    
    raw_data = pd.read_excel("C:\MyFiles\TEMP-ECE445\data\history.xls").fillna(0)
    
    model_input_buffer = []
    
    date_to_list = {}
    for id, row in raw_data.iterrows():
        time_string = row["time"]
        timestamp = datetime.strptime(time_string, '%d.%m.%Y %H:%M')
        date_to_list[f"{timestamp.month}-{timestamp.day}"] = {
            "temp_min": [],
            "temp_max": [],
            "pressure": [],
            "humidity": [],
            "wind_speed": [],
        }
    
    assert len(date_to_list.items()) == 48
    
    for id, row in raw_data.iterrows():
        time_string = row["time"]
        timestamp = datetime.strptime(time_string, '%d.%m.%Y %H:%M')
        date_to_list[f"{timestamp.month}-{timestamp.day}"]["temp_min"].append(float(row["Tn"]))
        date_to_list[f"{timestamp.month}-{timestamp.day}"]["temp_max"].append(float(row["Tx"]))
        date_to_list[f"{timestamp.month}-{timestamp.day}"]["pressure"].append(float(row["Po"]) * 133.32236842105 / 100)
        date_to_list[f"{timestamp.month}-{timestamp.day}"]["humidity"].append(float(row["U"]))
        date_to_list[f"{timestamp.month}-{timestamp.day}"]["wind_speed"].append(float(row["Ff"]))
    
    for date in date_to_list.keys():
        data_item = torch.tensor([
            min(remove_anomalies(date_to_list[date]["temp_min"])),
            max(remove_anomalies(date_to_list[date]["temp_max"])),
            avg(remove_anomalies(date_to_list[date]["pressure"])),
            avg(remove_anomalies(date_to_list[date]["humidity"])),
            avg(remove_anomalies(date_to_list[date]["wind_speed"]))
        ]).unsqueeze(0)
        model_input_buffer.append(data_item)
    
    assert len(model_input_buffer) == 48
    model_input_buffer.reverse()
    
    test_day_data = torch.cat(model_input_buffer, dim=0)
    assert test_day_data.shape == (48, 5)
    
    month_col = torch.tensor([float(datetime.now().month) for _ in range(48)]).unsqueeze(1)
    model_input = torch.cat((test_day_data, month_col), dim=1)
    assert model_input.shape == (48, 6)
    predictions = system.predict_multi_step(model_input, args.prediction_length)
    
    for i in range(len(predictions)):
        print(f"==> Future day {i+1}...")
        print(predictions[i])


if __name__ == '__main__':
    main()
    print("test.py: DONE!")
    open(os.path.join(args.out_root, "daily-DONE"), "w").close()
