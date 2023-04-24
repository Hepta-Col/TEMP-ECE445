import pdb
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from typing import List, Dict
from datetime import datetime

from common.config import *


class AutoRegressionDataset(Dataset):
    def __init__(self, csv_data: pd.DataFrame, historical_length: int) -> None:
        super().__init__()
        assert all(attrib in csv_data.columns for attrib in attributes_of_interest)

        temperature_list = csv_data['temp'].tolist()
        pressure_list = csv_data['pressure'].tolist()
        humidity_list = csv_data['humidity'].tolist()
        wind_speed_list = csv_data['wind_speed'].tolist()
        # rainfall_list = csv_data['rain_1h'].tolist()

        time_list = csv_data['dt_iso'].tolist()
        date_list = [time.split(' ')[0] for time in time_list]
        month_list = [datetime.strptime(date, '%Y-%m-%d').month for date in date_list]

        # x_list = [temperature_list, pressure_list, humidity_list, wind_speed_list, rainfall_list, month_list]
        # y_list = [temperature_list, pressure_list, humidity_list, wind_speed_list, rainfall_list]
        x_list = [temperature_list, pressure_list, humidity_list, wind_speed_list, month_list]
        y_list = [temperature_list, pressure_list, humidity_list, wind_speed_list, ]

        self.data: List[Dict[str, torch.Tensor]] = []

        start = 0
        end = historical_length
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

            start += historical_length
            end += historical_length
            
        print(f"data size: {len(self.data)}")

    def __getitem__(self, index):
        sample = self.data[index]
        return sample['x'], sample['y']

    def __len__(self) -> int:
        return len(self.data)


class PredictionDataset(Dataset):
    def __init__(self, csv_data: pd.DataFrame, historical_length: int, prediction_length: int) -> None:
        super().__init__()
        assert all(attrib in csv_data.columns for attrib in attributes_of_interest)

        temperature_list = csv_data['temp'].tolist()
        pressure_list = csv_data['pressure'].tolist()
        humidity_list = csv_data['humidity'].tolist()
        wind_speed_list = csv_data['wind_speed'].tolist()

        time_list = csv_data['dt_iso'].tolist()
        date_list = [time.split(' ')[0] for time in time_list]
        month_list = [datetime.strptime(date, '%Y-%m-%d').month for date in date_list]
        
        description_list = csv_data['weather_description']

        x_list = [temperature_list, pressure_list, humidity_list, wind_speed_list, month_list]
        y_list = [temperature_list, pressure_list, humidity_list, wind_speed_list, ]

        self.data: List[Dict[str, torch.Tensor]] = []

        start = 0
        end = historical_length
        num_time_steps = len(temperature_list)
        while start < num_time_steps - 1:
            if end > num_time_steps - 1:
                break

            x = torch.stack(
                [torch.as_tensor(l[start:end]) 
                 for l in x_list], dim=0).T
            y = torch.stack(
                [torch.as_tensor(l[end:end+prediction_length]) 
                 for l in y_list], dim=0).T
            d = torch.as_tensor([weather_descriptions.inverse[description_list[end+i]] for i in range(prediction_length)])

            self.data.append({'x': x, 'y': y, 'd': d})

            start += historical_length
            end += historical_length
            
        print(f"data size: {len(self.data)}")

    def __getitem__(self, index):
        sample = self.data[index]
        return sample['x'], sample['y'], sample['d']

    def __len__(self) -> int:
        return len(self.data)


def get_csv_data(csv_path):
    print(f"==> Reading csv from {csv_path}...")
    csv_data = pd.read_csv(csv_path)
    csv_data = csv_data.fillna(value=0) 
    return csv_data


def get_forecaster_training_dataloaders(csv_path, historical_length, train_test_ratio, batch_size):
    csv_data = get_csv_data(csv_path=csv_path)
    
    print(f"size of csv: {csv_data.shape[0]}")
    train_size = int((train_test_ratio / (train_test_ratio + 1)) * csv_data.shape[0])
    print("Making train set...")
    train_dataset = AutoRegressionDataset(csv_data[:train_size], historical_length)
    print("Making test set...")
    test_dataset = AutoRegressionDataset(csv_data[train_size:], historical_length)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_dataloader, test_dataloader


def get_system_evaluation_dataloader(csv_path, historical_length, prediction_length):
    csv_data = get_csv_data(csv_path=csv_path)
    
    print("Making evaluation set...")
    eval_dataset = PredictionDataset(csv_data, historical_length, prediction_length)

    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1, shuffle=False, drop_last=False)

    return eval_dataloader


def get_classifier_training_dataset(csv_path):
    csv_data = get_csv_data(csv_path=csv_path)
    
    time_list = csv_data['dt_iso'].tolist()
    date_list = [time.split(' ')[0] for time in time_list]
    month_list = [datetime.strptime(date, '%Y-%m-%d').month for date in date_list]
    month_col = pd.DataFrame({'month': month_list})
    
    csv_data = pd.concat([csv_data, month_col], axis=1)
    
    X_array = csv_data[names_for_input_features].values
    y_array = np.array([weather_descriptions.inverse[v] for v in csv_data['weather_description'].values])

    return {"X": X_array, "y": y_array}    
