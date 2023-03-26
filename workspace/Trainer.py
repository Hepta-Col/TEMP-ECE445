import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from WeatherDataset import get_weather_dataloaders
from models.Forecaster import Forecaster
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from common import *


class Trainer(object):
    def __init__(self, args) -> None:
        self.args = args
        
        self.train_loss_records = []
        self.test_loss_records = []

    def train(self):
        print("Training...")
        train_loss = eval_loss = 0
        with tqdm(total=self.args.num_epochs) as pbar:
            for epoch in range(self.args.num_epochs):
                pbar.set_description(f"Epoch {epoch}")
                
                train_loss = self._train()
                if self._eval_needed(epoch):
                    eval_loss = self._eval()
                
                pbar.set_postfix(train_loss=train_loss,eval_loss=eval_loss)
                pbar.update(1)

        self._plot_loss()

    def _train(self) -> float:
        """return average loss"""
        pass

    def _eval(self) -> float:
        """return average loss"""
        pass

    def _eval_needed(self, epoch) -> bool:
        return (epoch + 1) % self.args.eval_interval == 0
    
    def _avg(self, arr):
        return sum(arr) / len(arr)
    
    def _plot_loss(self):
        plt.figure()
        plt.plot(list(range(len(self.train_loss_records))), self.train_loss_records)
        plt.savefig("./train loss.jpg")
        
        plt.figure()
        plt.plot(list(range(len(self.test_loss_records))), self.test_loss_records)
        plt.savefig("./test loss.jpg")
        
    def _normalize(self, seq_data: torch.Tensor):
        """
        seq_data: [bs, seq_len, input size (6)]
        """
        ret = F.normalize(seq_data, p=2, dim=1)
        return ret


class ForecasterTrainer_V1(Trainer):
    def __init__(self, args) -> None:
        super().__init__(args)
        
        self.device = args.device
        
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
        self.forecaster = Forecaster(lstm_config, mlp_config).to(args.device)
        self.optimizer = torch.optim.AdamW(params=self.forecaster.parameters(), lr=args.lr)
        self.criterion = nn.MSELoss()

    def _train(self):
        self.forecaster.train()
        _records = []
        for batch_id, (seq_batch, tgt_batch) in enumerate(self.train_dataloader):
            seq_batch = self._normalize(seq_batch).to(self.device)
            tgt_batch = tgt_batch.to(self.device)
            pred_batch = self.forecaster(seq_batch)
            loss = self.criterion(pred_batch, tgt_batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            _records.append(loss.item())
        avg_loss = self._avg(_records)
        self.train_loss_records.append(avg_loss) 
        return avg_loss

    def _eval(self):
        self.forecaster.eval()
        _records = []
        with torch.no_grad():
            for batch_id, (seq_batch, tgt_batch) in enumerate(self.test_dataloader):
                seq_batch = self._normalize(seq_batch).to(self.device)
                tgt_batch = tgt_batch.to(self.device)
                
                pred_batch = self.forecaster(seq_batch)
                loss = self.criterion(pred_batch, tgt_batch)
                _records.append(loss.item())
        avg_loss = self._avg(_records)
        self.test_loss_records.append(avg_loss) 
        return avg_loss
