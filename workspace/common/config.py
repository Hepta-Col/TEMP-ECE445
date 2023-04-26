import os
import numpy as np
import torch
import random
from bidict import bidict
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


seed = 3407
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


now = datetime.now()
time = str(now.month) + "." + str(now.day) + "-" + str(now.hour) + ":" + str(now.minute) + ":" + str(now.second)

device = "cuda" if torch.cuda.is_available() else "cpu"

attributes_of_interest = ['dt', 'dt_iso', 'temp', 'pressure',
                          'humidity', 'wind_speed', 'weather_description']
names_for_input_features = ['temp', 'pressure', 'humidity', 'wind_speed', 'month']
names_for_output_features = ['temp', 'pressure', 'humidity', 'wind_speed', ]

weather_descriptions = bidict({0: 'few clouds', 
                               1: 'light rain', 
                               2: 'overcast clouds', 
                               3: 'sky is clear', 
                               4: 'light snow', 
                               5: 'broken clouds', 
                               6: 'scattered clouds', 
                               7: 'snow', 
                               8: 'moderate rain', 
                               9: 'heavy intensity rain', 
                               10: 'very heavy rain'})

#! input
csv_path = "../data/haining_weather.csv"
database_path =  "/root/qzt/TEMP-ECE445/data/weatherdata.db"

#! intermediate
forecaster_save_path = "../saved_models/forecaster.pt"
classifier_save_path = "../saved_models/classifier.pkl"

