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


def read_database(df, line_index):
    this_line: pd.Series = df.iloc[line_index]  
    time = this_line['time']
    time = datetime.strptime(time, '%m.%d_%H.%M.%S')
    base_timestamp = time.hour 
    
    try:
        next_line: pd.Series = df.iloc[line_index + 1]  
        time = next_line['time']
        time = datetime.strptime(time, '%m.%d_%H.%M.%S')
        next_timestamp = time.hour 
    except:
        next_timestamp = None
    
    return this_line, base_timestamp, next_timestamp


def append_stream(stream_buffer, line):
    temperature = float(line['temperature'])
    pressure = float(line['pressure']) / 100
    humidity = float(line['humidity'])
    wind_speed = float(line['wind'])

    dict_ = {'temperature': temperature,
            'pressure': pressure,
            'humidity': humidity,
            'wind_speed': wind_speed}
    
    stream_buffer.append(dict_)    
    

def transfer_stream_to_model_input(model_input_buffer, stream_buffer, now_month):
    temp_min, temp_max, pressure, humidity, wind_speed = compress(stream_buffer)
    stream_buffer.clear()

    data_item = torch.tensor([temp_min, 
                            temp_max, 
                            pressure, 
                            humidity, 
                            wind_speed, 
                            now_month]).unsqueeze(0)
    model_input_buffer.append(data_item) 
    

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


def do_prediction(system, model_input_buffer, time_record=None):
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

    if time_record:
        time_temp = time_record
    else:
        time_temp = tm.localtime(tm.time())
        time_temp = "{0}.{1}_{2:0>2d}.{3:0>2d}.{4:0>2d}".format(time_temp[1],time_temp[2],time_temp[3],time_temp[4],time_temp[5])

    s = [time_temp]
    s += [str(round(predictions[k].temp_min.item(), 2)) for k in range(args.prediction_length)]
    s += [str(round(predictions[k].temp_max.item(), 2)) for k in range(args.prediction_length)]
    s += [str(round(predictions[k].humidity.item(), 2)) for k in range(args.prediction_length)]
    s += [str(round(predictions[k].pressure.item(), 2)) for k in range(args.prediction_length)]
    s += [str(round(predictions[k].wind_speed.item(), 2)) for k in range(args.prediction_length)]
    s += [predictions[k].description for k in range(args.prediction_length)]

    print("==> Writing predictions...")
    write_predictions(s)
    s.clear()


def main():
    system = System(args)
    now_month = float(datetime.now().month)
    
    stream_buffer = []
    model_input_buffer = []
    base_timestamp = datetime.now().hour
    num_lines = 0
    
    try:
        with sql.connect(args.database_path) as con:
            df: pd.DataFrame = pd.read_sql("SELECT * FROM weatherdata", con=con)
            df = df.fillna(value=0) 
            num_lines = df.shape[0]
            assert num_lines > 2
            print(f"==> Database detected. Size: {num_lines}")
            
            print("==> Filling model input buffer with existing database...")
            line_index = 0     #! <----
            while True:                
                # print(f"Reading line index {line_index}...")
                line, base_timestamp, next_timestamp = read_database(df, line_index)
                append_stream(stream_buffer, line)
                
                if line_index >= num_lines - 1:
                    break
                
                if next_timestamp != base_timestamp:
                    assert next_timestamp == (base_timestamp + 1) % 24
                    transfer_stream_to_model_input(model_input_buffer, stream_buffer, now_month)
                    
                    if len(model_input_buffer) == args.historical_length:
                        do_prediction(system, model_input_buffer, line['time'])
                        model_input_buffer = model_input_buffer[1:]
                
                line_index += 1     #! <----
        print(f"==> Buffers: stream buffer: {len(stream_buffer)}; model input buffer: {len(model_input_buffer)}")
    except:
        pdb.set_trace()
        pass

    print("\n=================================================")
    print("REAL TIME PREDICTION")
    print("=================================================\n")

    it = 0
    while True:
        it = (it + 1) % 6
        now_month = float(datetime.now().month)
        with sql.connect(args.database_path) as con:
            df: pd.DataFrame = pd.read_sql("SELECT * FROM weatherdata", con=con)
            df = df.fillna(value=0)
            if df.shape[0] == num_lines:
                print(f"==> Waiting for new data" + "." * it, "\r", end='')
                tm.sleep(0.05)
                continue
            
            print("\n==> New data comes in!")
            num_lines = df.shape[0]
            new_line, new_timestamp, _ = read_database(df, -1)
            print(new_line)
            print(f"==> Buffers: stream buffer: {len(stream_buffer)}; model input buffer: {len(model_input_buffer)}")
            
            append_stream(stream_buffer, new_line)

            if new_timestamp != base_timestamp and len(stream_buffer) > 2:
                print(f"==> Next timestamp. Length of stream buffer: {len(stream_buffer)}")
                base_timestamp = new_timestamp
                
                transfer_stream_to_model_input(model_input_buffer, stream_buffer, now_month)
                
                if len(model_input_buffer) == args.historical_length:
                    do_prediction(system, model_input_buffer)
                    model_input_buffer = model_input_buffer[1:]
                    

if __name__ == '__main__':
    main()
    print("run.py: DONE!")
    open(os.path.join(args.out_root, "run-DONE"), "w").close()
