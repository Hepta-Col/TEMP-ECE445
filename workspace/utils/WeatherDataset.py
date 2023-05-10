import pdb
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from typing import List, Dict
from datetime import datetime
from tqdm import tqdm, trange

from common.config import *


def _transform_list(input_list, type: str, interval: int = 24):
    input_arr = np.array(input_list)
    
    ret = []
    start = 0
    end = interval
    
    while end <= len(input_arr):
        group = input_arr[start:end]
        if type == 'avg':   
            ret.append(group.mean())
        elif type == 'min':
            ret.append(group.min())
        elif type == 'max':
            ret.append(group.max())
        elif type == 'first':
            ret.append(group[0])
        
        start += interval
        end += interval

    return ret


class AutoRegressionDataset(Dataset):
    def __init__(self, csv_data: pd.DataFrame, historical_length: int, granularity: str) -> None:
        super().__init__()
        assert all(attrib in csv_data.columns for attrib in attributes_of_interest)

        temp_min_list = csv_data['temp_min'].tolist()
        if granularity == 'day':
            temp_min_list = _transform_list(temp_min_list, 'min')
            
        temp_max_list = csv_data['temp_max'].tolist()
        if granularity == 'day':
            temp_max_list = _transform_list(temp_max_list, 'max')
            
        pressure_list = csv_data['pressure'].tolist()
        if granularity == 'day':
            pressure_list = _transform_list(pressure_list, 'avg')
            
        humidity_list = csv_data['humidity'].tolist()
        if granularity == 'day':
            humidity_list = _transform_list(humidity_list, 'avg')        
        
        wind_speed_list = csv_data['wind_speed'].tolist()
        if granularity == 'day':
            wind_speed_list = _transform_list(wind_speed_list, 'avg') 

        time_list = csv_data['dt_iso'].tolist()
        date_list = [time.split(' ')[0] for time in time_list]
        month_list = [datetime.strptime(date, '%Y-%m-%d').month for date in date_list]
        if granularity == 'day':
            month_list = _transform_list(month_list, 'first') 

        x_list = [temp_min_list, temp_max_list, pressure_list, humidity_list, wind_speed_list, month_list]
        y_list = [temp_min_list, temp_max_list, pressure_list, humidity_list, wind_speed_list, ]

        self.data: List[Dict[str, torch.Tensor]] = []
        
        start = 0
        end = historical_length
        num_time_steps = len(temp_min_list)
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
        return sample['x'].float(), sample['y'].float()

    def __len__(self) -> int:
        return len(self.data)


class PredictionDataset(Dataset):
    def __init__(self, csv_data: pd.DataFrame, historical_length: int, prediction_length: int, granularity: str) -> None:
        super().__init__()
        assert all(attrib in csv_data.columns for attrib in attributes_of_interest)

        temp_min_list = csv_data['temp_min'].tolist()
        if granularity == 'day':
            temp_min_list = _transform_list(temp_min_list, 'min')
            
        temp_max_list = csv_data['temp_max'].tolist()
        if granularity == 'day':
            temp_max_list = _transform_list(temp_max_list, 'max')
            
        pressure_list = csv_data['pressure'].tolist()
        if granularity == 'day':
            pressure_list = _transform_list(pressure_list, 'avg')
            
        humidity_list = csv_data['humidity'].tolist()
        if granularity == 'day':
            humidity_list = _transform_list(humidity_list, 'avg')        
        
        wind_speed_list = csv_data['wind_speed'].tolist()
        if granularity == 'day':
            wind_speed_list = _transform_list(wind_speed_list, 'avg') 

        time_list = csv_data['dt_iso'].tolist()
        date_list = [time.split(' ')[0] for time in time_list]
        month_list = [datetime.strptime(date, '%Y-%m-%d').month for date in date_list]
        if granularity == 'day':
            month_list = _transform_list(month_list, 'first') 
        
        description_list = csv_data['weather_description']

        x_list = [temp_min_list, temp_max_list, pressure_list, humidity_list, wind_speed_list, month_list]
        y_list = [temp_min_list, temp_max_list, pressure_list, humidity_list, wind_speed_list, ]

        self.data: List[Dict[str, torch.Tensor]] = []

        start = 0
        end = historical_length
        num_time_steps = len(temp_min_list)
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
        return sample['x'].float(), sample['y'].float(), sample['d'].float()

    def __len__(self) -> int:
        return len(self.data)


def get_csv_data(csv_path):
    print(f"==> Reading csv from {csv_path}...")
    csv_data = pd.read_csv(csv_path)
    csv_data = csv_data.fillna(value=0) 
    return csv_data


def get_forecaster_training_dataloaders(csv_path, historical_length, train_test_ratio, batch_size, granularity):
    csv_data = get_csv_data(csv_path=csv_path)
    
    print(f"size of csv: {csv_data.shape[0]}")
    train_size = int((train_test_ratio / (train_test_ratio + 1)) * csv_data.shape[0])
    print("Making train set...")
    train_dataset = AutoRegressionDataset(csv_data[:train_size], historical_length, granularity)
    print("Making test set...")
    test_dataset = AutoRegressionDataset(csv_data[train_size:], historical_length, granularity)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_dataloader, test_dataloader


def get_system_evaluation_dataloader(csv_path, historical_length, prediction_length, granularity):
    csv_data = get_csv_data(csv_path=csv_path)
    
    print("Making evaluation set...")
    eval_dataset = PredictionDataset(csv_data, historical_length, prediction_length, granularity)

    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1, shuffle=False, drop_last=False)

    return eval_dataloader


def get_classifier_training_dataset(csv_path, new_csv_path, train_test_ratio):
    def preprocess_table(csv_path, new_csv_path):
        if os.path.exists(new_csv_path):
            return pd.read_csv(new_csv_path)
    
        csv_data = get_csv_data(csv_path=csv_path)

        def reduce_description(description: str):
            # if 'cloud' in description:
            #     return 'cloudy'
            # elif 'rain' in description:
            #     return 'rainy'
            # elif 'snow' in description:
            #     return 'snowy'
            # else:
            #     return 'sunny'
            
            if 'rain' in description or 'snow' in description:
                return 'rainy'
            else:
                return 'sky is clear'
        
        time_list = csv_data['dt_iso'].tolist()
        date_list = [time.split(' ')[0] for time in time_list]
        month_list = [datetime.strptime(date, '%Y-%m-%d').month for date in date_list]
        month_col = pd.DataFrame({'month': month_list})
        
        csv_data = pd.concat([csv_data, month_col], axis=1)
        
        print("==> Reducing...")
        for i in trange(csv_data.shape[0]):
            try:
                csv_data.loc[i, 'weather_description'] = reduce_description(csv_data.loc[i, 'weather_description'])
            except:
                pdb.set_trace()   
        
        print("==> Saving csv...")
        csv_data.to_csv(new_csv_path)
        
        return csv_data

    def get_X_and_y(csv_data):     
        X_array = csv_data[names_for_input_features].values
        y_array = np.array([weather_descriptions.inverse[v] for v in csv_data['weather_description'].values])
        return X_array, y_array

    csv_data = preprocess_table(csv_path, new_csv_path)
    print(f"size of csv: {csv_data.shape[0]}")
    train_size = int((train_test_ratio / (train_test_ratio + 1)) * csv_data.shape[0])
    train_data, test_data = csv_data[:train_size].reset_index(), csv_data[train_size:].reset_index()
    print("==> Making train data...")
    train_X_array, train_y_array = get_X_and_y(train_data)
    print("==> Making test data...")
    test_X_array, test_y_array = get_X_and_y(test_data)

    return {"X": train_X_array, "y": train_y_array}, {"X": test_X_array, "y": test_y_array}