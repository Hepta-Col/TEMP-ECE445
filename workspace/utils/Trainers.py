import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.WeatherDataset import get_forecaster_training_dataloaders, get_classifier_training_dataset
from models.Forecaster import Forecaster
from models.Classifier import Classifier
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import pickle as pkl
from common.common import *


class NaiveTrainer(object):
    def __init__(self, args) -> None:
        self.args = args

    def _avg(self, arr):
        return sum(arr) / len(arr)
    
    def train(self):
        pass


class DNNTrainer(NaiveTrainer):
    def __init__(self, args) -> None:
        super().__init__(args)
        
        self.train_loss_records = []
        self.test_loss_records = []

    def train(self):
        print("==> Training DNN...")
        train_loss = eval_loss = 0
        with tqdm(total=self.args.num_epochs) as pbar:
            for epoch in range(self.args.num_epochs):
                pbar.set_description(f"Epoch {epoch}")
                
                train_loss, model = self._train()
                if self._test_needed(epoch):
                    eval_loss = self._test()
                
                pbar.set_postfix(train_loss=train_loss,eval_loss=eval_loss)
                pbar.update(1)

        print("==> Saving model...")
        torch.save(model.state_dict(), forecaster_save_path)

        """Loading model:
        model.load_state_dict(torch.load(forecaster_save_path))
        """

        print("==> Plotting...")
        self._plot_loss()

    def _train(self):
        """return average loss"""
        pass

    def _test(self) -> float:
        """return average loss"""
        pass

    def _test_needed(self, epoch) -> bool:
        return (epoch + 1) % self.args.eval_interval == 0
    
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


class ForecasterTrainer_V1(DNNTrainer):
    def __init__(self, args) -> None:
        super().__init__(args)
        
        self.device = device
        
        self.train_dataloader, self.test_dataloader = get_forecaster_training_dataloaders(
            csv_path, sequence_length, args.train_test_ratio, args.batch_size)

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
        self.forecaster = Forecaster(lstm_config, mlp_config).to(device)
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
        return avg_loss, self.forecaster

    def _test(self):
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


class ClassifierTrainer_V1(NaiveTrainer):
    def __init__(self, args) -> None:
        super().__init__(args)
        
        self.training_data = get_classifier_training_dataset(csv_path)
        self.classifier = Classifier()
    
    def train(self):
        print("==> Training classifier...")
        self.classifier.fit(X=self.training_data["X"], y=self.training_data["y"])

        print("==> Saving model...")
        with open(classifier_save_path, 'wb') as f:
            pkl.dump(self.classifier, f)

        """Loading model:
        with open(pkl_filename, 'rb') as file:
            pickle_model = pickle.load(file)
        """
        
        print("==> Visualizing tree...")
        self.classifier.visualize()
