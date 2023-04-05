device = "cuda"

sequence_length = 24

attributes_of_interest = ['dt', 'dt_iso', 'temp', 'pressure',
                          'humidity', 'wind_speed', 'rain_1h', 'weather_description']
names_for_input_features = ['temperature', 'pressure', 'humidity', 'wind_speed', 'rainfall', 'month']
names_for_output_features = ['temperature', 'pressure', 'humidity', 'wind_speed', 'rainfall']

csv_path = "../data/haining_weather.csv"
save_path = "./saved_model/model.pt"