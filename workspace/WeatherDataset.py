import pdb
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm, trange
from typing import List, Dict
from common import attributes_of_interest, measures


class AutoRegressionDataset(Dataset):
    def __init__(self, args, measure_of_interest) -> None:
        super().__init__()

        self.args = args
        self.measure_of_interest = measure_of_interest

        self.csv_data: pd.DataFrame = pd.read_csv(args.csv_path)
        assert all(
            attrib in self.csv_data.columns for attrib in attributes_of_interest)
        self.data_of_interest: pd.DataFrame = self.csv_data[measure_of_interest]
        self.auto_regressive_data = self._generate_sequences()

    def __getitem__(self, index) -> Dict:
        sample = self.auto_regressive_data[index]
        return torch.Tensor(sample['seq']), torch.Tensor(sample['tgt'])

    def __len__(self) -> int:
        return len(self.auto_regressive_data)

    def _generate_sequences(self) -> List[Dict[str, List]]:
        ret = []

        print("==> Generating sequences...")
        for idx in trange(self.data_of_interest.shape[0] - self.args.look_back_window):
            seq = self.data_of_interest[idx:idx +
                                        self.args.look_back_window].values
            tgt = self.data_of_interest[idx + self.args.look_back_window:idx +
                                        self.args.look_back_window + self.args.prediction_window].values
            ret.append({'seq': seq, 'tgt': tgt})

        return ret


class ClassificationDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()

    def __getitem__(self, index):
        return super().__getitem__(index)
    
    def __len__():
        pass