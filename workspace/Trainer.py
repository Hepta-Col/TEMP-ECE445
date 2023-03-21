import torch
import torch.nn as nn
from WeatherDataset import get_weather_dataloaders
from models.Forecaster import Forecaster
from tqdm import tqdm
from common import *


class Trainer(object):
    def __init__(self, args) -> None:
        self.args = args

    def train(self):
        for epoch in range(self.args.num_epochs):
            print(f"EPOCH {epoch} ---------------------------------------------------")
            print("Training...")
            self._train()
            if self._eval_needed(epoch):
                print("Evaluating...")
                self._eval()

    def _train(self):
        pass

    def _eval(self):
        pass

    def _eval_needed(self, epoch) -> bool:
        return (epoch + 1) % self.args.eval_interval == 0


class ForecasterTrainer_V1(Trainer):
    def __init__(self, args) -> None:
        super().__init__(args)
        
        self.train_dataloader, self.test_dataloader = get_weather_dataloaders(
            args.csv_path, args.sequence_length, args.train_test_ratio, args.batch_size)

        lstm_config = {
            'input_size': len(names_for_input_features),
            'hidden_size': args.lstm_hidden_size,
            'num_layers': args.lstm_num_layers,
            'dropout': args.lstm_dropout,
            'batch_first': True,
        }
        mlp_config = {
            'num_layers': args.mlp_num_layers,
            'input_size': args.lstm_hidden_size,
            'hidden_size': args.mlp_hidden_size,
            'output_size': len(names_for_output_features),
            'dropout': args.mlp_dropout,
        }
        self.forecaster = Forecaster(lstm_config, mlp_config)
        self.optimizer = torch.optim.AdamW(params=self.forecaster.parameters(), lr=args.lr)
        self.criterion = nn.MSELoss()

        self.train_loss_records = []
        self.test_loss_records = []

    def _train(self):
        self.forecaster.train()
        for batch_id, (seq_batch, tgt_batch) in tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader)):
            pred_batch = self.forecaster(seq_batch)
            loss = self.criterion(pred_batch, tgt_batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.train_loss_records.append(loss.item())

    def _eval(self):
        self.forecaster.eval()
        with torch.no_grad():
            for batch_id, (seq_batch, tgt_batch) in tqdm(enumerate(self.test_dataloder), total=len(self.test_dataloder)):
                pred_batch = self.forecaster(seq_batch)
                loss = self.criterion(pred_batch, tgt_batch)
                self.test_loss_records.append(loss.item())
