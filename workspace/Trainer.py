from utils import get_weather_dataloader
from Forecaster import Forecaster


class Trainer(object):
    def __init__(self, args) -> None:
        self.args = args
        
        self.train_dataloader = get_weather_dataloader(args, type='train')
        self.test_dataloader = get_weather_dataloader(args, type='test')
        self.forecaster = Forecaster(args)

    def train(self):
        pass