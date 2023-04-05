from bidict import bidict


device = "cuda"

sequence_length = 24

attributes_of_interest = ['dt', 'dt_iso', 'temp', 'pressure',
                          'humidity', 'wind_speed', 'rain_1h', 'weather_description']
names_for_input_features = ['temp', 'pressure', 'humidity', 'wind_speed', 'rain_1h', 'month']
names_for_output_features = ['temp', 'pressure', 'humidity', 'wind_speed', 'rain_1h']

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
forecaster_save_path = "../out/saved_models/forecaster.pt"
classifier_save_path = "../out/saved_models/classifier.pkl"
figs_dir = "../out/figs"
