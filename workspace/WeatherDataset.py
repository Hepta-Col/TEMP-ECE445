import pdb
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from typing import List, Dict
from datetime import datetime
from common import *


class AutoRegressionDataset(Dataset):
    def __init__(self, csv_path, sequence_length) -> None:
        super().__init__()

        csv = pd.read_csv(csv_path)
        csv = csv.fillna(value=0)
        assert all(attrib in csv.columns for attrib in attributes_of_interest)

        temperature_list = csv['temp'].tolist()
        pressure_list = csv['pressure'].tolist()
        humidity_list = csv['humidity'].tolist()
        wind_speed_list = csv['wind_speed'].tolist()
        rainfall_list = csv['rain_1h'].tolist()

        time_list = csv['dt_iso'].tolist()
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
                end = num_time_steps - 1

            x = torch.stack(
                [torch.as_tensor(l[start:end]) 
                 for l in x_list], dim=0).T
            y = torch.stack(
                [torch.as_tensor(l[start+1:end+1]) 
                 for l in y_list], dim=0).T

            self.data.append({'x': x, 'y': y})

            start += sequence_length
            end += sequence_length

    def __getitem__(self, index):
        sample = self.data[index]
        return sample['x'], sample['y']

    def __len__(self) -> int:
        return len(self.data)


def get_weather_dataloaders(csv_path, sequence_length, train_test_ratio, batch_size):
    dataset = AutoRegressionDataset(csv_path, sequence_length)
    train_size = int((train_test_ratio / (train_test_ratio + 1)) * len(dataset))
    train_dataset = dataset[:train_size]
    test_dataset = dataset[train_size:]

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_dataloader, test_dataloader
