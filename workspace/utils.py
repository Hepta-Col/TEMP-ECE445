from random import shuffle
from torch.utils.data import Dataset, DataLoader

from WeatherDataset import WeatherDataset


def get_weather_dataloader(args, type):
    dataset = WeatherDataset(args)
    train_size = int((args.train_test_ratio / (args.train_test_ratio + 1)) * len(dataset))
    shuffle(dataset)
    train_dataset = dataset[:train_size]
    test_dataset = dataset[train_size:]

    if type == 'train':
        return DataLoader(dataset=train_dataset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        drop_last=True)
    elif type == 'test':
        return DataLoader(dataset=test_dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        drop_last=True)
