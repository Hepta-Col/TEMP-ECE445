import pdb
import os
import torch
import pandas as pd
import time as tm
import sqlite3 as sql
from common.args import args
from common.config import *
from common.funcs import *
from utils.System import System
from demo.write_sql2_start import write_predictions


def compress(buffer):
    temperature_list = [g['temperature'] for g in buffer]        
    pressure_list = [g['pressure'] for g in buffer]        
    humidity_list = [g['humidity'] for g in buffer]        
    wind_speed_list = [g['wind_speed'] for g in buffer]        
    
    temp_min = min(remove_anomalies(temperature_list))
    temp_max = max(remove_anomalies(temperature_list))
    pressure = avg(remove_anomalies(pressure_list))
    humidity = avg(remove_anomalies(humidity_list))
    wind_speed = avg(remove_anomalies(wind_speed_list))
    
    return float(temp_min), float(temp_max), float(pressure), float(humidity), float(wind_speed)


def main():
    system = System(args)
    
    it = 0
    num_lines = 0
    stream_buffer = []
    model_input_buffer = []
    base_timestamp = datetime.now().minute
    
    while True:
        it = (it + 1) % 6
        now_month = float(datetime.now().month)
        with sql.connect(args.database_path) as con:
            df: pd.DataFrame = pd.read_sql("SELECT * FROM weatherdata", con=con)
            if df.shape[0] == num_lines:
                print(f"==> Waiting for new data" + "." * it, "\r", end='')
                tm.sleep(0.05)
                continue
            
            print("\n==> New data comes in!")
            num_lines = df.shape[0]
            
            last_line: pd.Series = df.iloc[-1]
            print(last_line)
            print(f"==> Buffers: stream buffer: {len(stream_buffer)}; model input buffer: {len(model_input_buffer)}")
            
            time = last_line['time']
            time = datetime.strptime(time, '%m.%d_%H.%M.%S')
            
            temperature = float(last_line['temperature'])
            pressure = float(last_line['pressure']) / 100
            humidity = float(last_line['humidity'])
            wind_speed = float(last_line['wind'])

            dict_ = {'temperature': temperature,
                     'pressure': pressure,
                     'humidity': humidity,
                     'wind_speed': wind_speed}
            
            stream_buffer.append(dict_)

            if time.minute != base_timestamp and len(stream_buffer) > 2:
                print(f"==> Next timestamp. Length of stream buffer: {len(stream_buffer)}")
                base_timestamp = time.minute
                
                temp_min, temp_max, pressure, humidity, wind_speed = compress(stream_buffer)
                print("==> Clearing stream buffer...")
                stream_buffer.clear()

                data_item = torch.tensor([temp_min, 
                                          temp_max, 
                                          pressure, 
                                          humidity, 
                                          wind_speed, 
                                          now_month]).unsqueeze(0)
                model_input_buffer.append(data_item)
                
                if len(model_input_buffer) == args.historical_length:
                    print("==> Model input buffer full")
                    print("==> Model input:")
                    model_input = torch.cat(model_input_buffer, dim=0)
                    print(model_input)
                    print("==> Making predictions...")
                    predictions = system.predict_multi_step(model_input, args.prediction_length)
                    assert len(predictions) == args.prediction_length
                    for i in range(args.prediction_length):
                        print(f"==> Future step {i}:")
                        print(predictions[i])
                    print("==> Clearing model input buffer...")
                    model_input_buffer.clear()
                    
                    s = [tm.localtime(tm.time())]
                    s += [predictions[k].temp_min for k in range(args.prediction_length)]
                    s += [predictions[k].temp_max for k in range(args.prediction_length)]
                    s += [predictions[k].humidity for k in range(args.prediction_length)]
                    s += [predictions[k].pressure for k in range(args.prediction_length)]
                    s += [predictions[k].wind_speed for k in range(args.prediction_length)]

                    print("==> Writing predictions...")
                    write_predictions(s)
                    s.clear()


if __name__ == '__main__':
    main()
    print("run.py: DONE!")
    open(os.path.join(args.out_root, "run-DONE"), "w").close()
