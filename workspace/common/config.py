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
time = str(now.month) + "." + str(now.day) + "-" + str(now.hour) + "." + str(now.minute) + "." + str(now.second)

device = "cuda" if torch.cuda.is_available() else "cpu"

attributes_of_interest = ['dt_iso', 'temp_min', 'temp_max', 'pressure', 'humidity', 'wind_speed', 'weather_description']
names_for_input_features =  ['temp_min', 'temp_max', 'pressure', 'humidity', 'wind_speed', 'month']
names_for_output_features = ['temp_min', 'temp_max', 'pressure', 'humidity', 'wind_speed', ]

# weather_descriptions = bidict({0: 'few clouds', 
#                                1: 'light rain', 
#                                2: 'overcast clouds', 
#                                3: 'sky is clear', 
#                                4: 'light snow', 
#                                5: 'broken clouds', 
#                                6: 'scattered clouds', 
#                                7: 'snow', 
#                                8: 'moderate rain', 
#                                9: 'heavy intensity rain', 
#                                10: 'very heavy rain'})

# weather_descriptions = bidict({0: 'cloudy',
#                                1: 'rainy',
#                                2: 'sunny',
#                                3: 'snowy'})

weather_descriptions = bidict({0: 'rainy', 1: 'sky is clear'})