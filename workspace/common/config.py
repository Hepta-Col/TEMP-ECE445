import os
import torch
from bidict import bidict
from datetime import datetime


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

csv_path = "../data/haining_weather.csv"
forecaster_save_path = "../saved_models/forecaster.pt"
classifier_save_path = "../saved_models/classifier.pkl"

out_root = os.path.join("../out", time)

figs_dir = os.path.join(out_root, "figs")
logs_dir = os.path.join(out_root, "logs")

os.makedirs(figs_dir)
os.makedirs(logs_dir)

args_path = os.path.join(out_root, "args.json")