import pdb
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm, trange
from typing import List, Dict


attributes_of_interest = ['dt', 'temp', 'pressure',
                          'humidity', 'wind_speed', 'weather_description']
measures = ['temp', 'pressure', 'humidity', 'wind_speed']


class WeatherDataset(Dataset):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.csv_data: pd.DataFrame = pd.read_csv(args.csv_path)
        assert all(
            attrib in self.csv_data.columns for attrib in attributes_of_interest)
        self.data_of_interest: pd.DataFrame = self.csv_data[attributes_of_interest]
        self.auto_regressive_data = self._generate_sequences()

    def __getitem__(self, index) -> Dict:
        return self.auto_regressive_data[index]

    def __len__(self) -> int:
        return len(self.auto_regressive_data)

    def _generate_sequences(self) -> List[Dict[str, Dict[str, torch.Tensor]]]:
        """return auto_regressive_data:
        [
            {
                'temp': {
                    'seq': [1, 2, 3],
                    'tgt': [4]
                }
                'pressure': {
                    'seq": [...],
                    'tgt': [...]
                },
                ...
            }
        ]
        """
        ret = []
        for i in trange(self.data_of_interest.shape[0] - self.args.look_back_window):
            out_dict = {}
            for m in measures:
                in_dict = {}
                in_dict['seq'] = torch.Tensor(self.data_of_interest[m][i: i +
                                                          self.args.look_back_window].values)
                in_dict['tgt'] = torch.Tensor(self.data_of_interest[m][i + self.args.look_back_window: i +
                                                          self.args.look_back_window + self.args.predict_window].values)
                out_dict[m] = in_dict
            ret.append(out_dict)
        return ret
