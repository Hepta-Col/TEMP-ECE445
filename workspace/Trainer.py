import torch
import torch.nn as nn
from utils import get_weather_dataloader
from Model import Forecaster
from common import measures
from tqdm import tqdm


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

    def _eval_needed(self, epoch):
        pass


class ForecasterTrainer(Trainer):
    def __init__(self, args) -> None:
        super().__init__(args)
        
        self.train_dataloaders = {m: get_weather_dataloader(args, type='train', measure_of_interest=m) for m in measures}
        self.test_dataloader = {m: get_weather_dataloader(args, type='test', measure_of_interest=m) for m in measures}

        self.forecasters = {m: Forecaster(args) for m in measures}
        self.optimizers = {m: torch.optim.AdamW(params=self.forecasters[m].parameters(), lr=args.lr) for m in measures}

        self.criterion = nn.MSELoss()

        self.train_loss_records = {m: [] for m in measures}
        self.test_loss_records = {m: [] for m in measures}

    def _train(self):
        for m in measures:
            train_dataloader = self.train_dataloaders[m]
            forecaster = self.forecasters[m].train()
            optimizer = self.optimizers[m]
            train_loss_record = self.train_loss_records[m]

            for batch_id, (seq_batch, tgt_batch) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
                pred_batch = forecaster(seq_batch)
                loss = self.criterion(pred_batch, tgt_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss_record.append(loss.item())

    def _eval(self):
        for m in measures:
            test_dataloder = self.test_dataloader[m]
            forecaster = self.forecasters[m].eval()
            test_loss_record = self.test_loss_records[m]

            for batch_id, (seq_batch, tgt_batch) in tqdm(enumerate(test_dataloder), total=len(test_dataloder)):
                pred_batch = forecaster(seq_batch)
                loss = self.criterion(pred_batch, tgt_batch)
                test_loss_record.append(loss.item())

    def _eval_needed(self, epoch):
        return (epoch + 1) % self.args.eval_interval == 0


class ClassifierTrainer(Trainer):
    def __init__(self, args) -> None:
        super().__init__(args)

    def _train(self):
        return super()._train()
    
    def _eval(self):
        return super()._eval()

    def _eval_needed(self, epoch):
        return super()._eval_needed(epoch)