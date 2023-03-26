import pdb
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from typing import List, Dict
from datetime import datetime
from common import *


class AutoRegressionDataset(Dataset):
    def __init__(self, csv_data: pd.DataFrame, sequence_length: int) -> None:
        super().__init__()
        assert all(attrib in csv_data.columns for attrib in attributes_of_interest)

        temperature_list = csv_data['temp'].tolist()
        pressure_list = csv_data['pressure'].tolist()
        humidity_list = csv_data['humidity'].tolist()
        wind_speed_list = csv_data['wind_speed'].tolist()
        rainfall_list = csv_data['rain_1h'].tolist()

        time_list = csv_data['dt_iso'].tolist()
        date_list = [time.split(' ')[0] for time in time_list]
        month_list = [datetime.strptime(date, '%Y-%m-%d').month for date in date_list]

        x_list = [temperature_list, pressure_list, humidity_list, wind_speed_list, rainfall_list, month_list]
        y_list = [temperature_list, pressure_list, humidity_list, wind_speed_list, rainfall_list]

        self.data: List[Dict[str, torch.Tensor]] = []

        start = 0
        end = sequence_length
        num_time_steps = len(temperature_list)
        while start < num_time_steps - 1:
            if end > num_time_steps - 1:
                break

            x = torch.stack(
                [torch.as_tensor(l[start:end]) 
                 for l in x_list], dim=0).T
            y = torch.stack(
                [torch.as_tensor(l[start+1:end+1]) 
                 for l in y_list], dim=0).T

            self.data.append({'x': x, 'y': y})

            start += sequence_length
            end += sequence_length
            
        print(f"data size: {len(self.data)}")

    def __getitem__(self, index):
        sample = self.data[index]
        return sample['x'], sample['y']

    def __len__(self) -> int:
        return len(self.data)


def get_weather_dataloaders(csv_path, sequence_length, train_test_ratio, batch_size):
    print(f"==> Reading csv from {csv_path}...")
    csv_data = pd.read_csv(csv_path)
    csv_data = csv_data.fillna(value=0)
    
    print(f"size of csv: {csv_data.shape[0]}")
    train_size = int((train_test_ratio / (train_test_ratio + 1)) * csv_data.shape[0])
    print("Making train set...")
    train_dataset = AutoRegressionDataset(csv_data[:train_size], sequence_length)
    print("Making test set...")
    test_dataset = AutoRegressionDataset(csv_data[train_size:], sequence_length)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_dataloader, test_dataloader
