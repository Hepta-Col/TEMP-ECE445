import sqlite3
import time
import datetime
from statistics import mean 

from common.args import args


def time_subtract(cur_time, timestamp):
    time_split = timestamp.split("_")
    time_test = datetime.datetime(year = 2023, month = int(time_split[0].split(".")[0]), day = int(time_split[0].split(".")[1]),hour = int(time_split[1].split(".")[0]), minute = int(time_split[1].split(".")[1]), second = int(time_split[1].split(".")[2]))
    timedelta = (cur_time-time_test).seconds
    return timedelta


def get_mean(records):
    temp = 0
    humi = 0
    pres = 0
    rain = 0
    wind = 0
    uv = 0
    pm1 = 0
    pm2_5 = 0
    pm10 = 0
    
    latest_time = records[-1][1]

    for i in range(len(records)):
        temp += float(records[i][2])
        humi += float(records[i][3])
        pres += float(records[i][4])
        rain += float(records[i][5])
        wind += float(records[i][6])
        uv   += float(records[i][7])
        pm1  += float(records[i][8])
        pm2_5+= float(records[i][9])
        pm10 += float(records[i][10])

    if_rain = 1 if rain > 0 else 0

    dict_ = {'day': latest_time.split("_")[0], 'time': latest_time.split("_")[1][:-3], 'temp':round(temp/len(records)), 'humi':round(humi/len(records)), 
            'pres':round(pres/(100*len(records))), 'rain': if_rain, 'wind': round(wind/len(records), 2), 'uv': round(uv/len(records)),
            'pm10': round(pm1/len(records), 1), 'pm25': round(pm2_5/len(records), 1), 'pm100': round(pm10/len(records), 1)}
    return dict_


def retrieve_record(min):
    """fetch newest data (unit: minute)"""
    
    time_check = datetime.datetime.now()
    print("Now the time is: ", time_check)

    conn = sqlite3.connect(args.database_path)
    print(f"Connected to database: {args.database_path}")
    
    cur = conn.cursor()
    query = "SELECT max(rowid) from weatherdata"
    cursor = cur.execute(query)
    max_line = cursor.fetchone()[0]
    print("Database max line: ", max_line)   

    query = """SELECT * FROM weatherdata WHERE rowid > ?"""
    cursor = cur.execute(query, (max_line-15*min,))
    records = cursor.fetchall()
    
    print(records)
    print("\n")
    cur.close()   

    while 1:
        if len(records)!= 0 and time_subtract(time_check, records[0][1]) <= 60*min:
            break
        else:
            if len(records) != 0:
                records.pop(0)
            else:
                break
    print(records)

    # cursor = cur.execute("Select * from (select rownum no ,serv_id from serv_history_517 ) where no>10")

    print ("Records edited")
    conn.close()

    if len(records) == 0:
        return 
    
    mean_record = get_mean(records)
    """格式
    1286, '5.8_20.34.02', '24.10', '56.00', '101863', '0', '4.11', '0', '18', '25', '26'
    行数(id)，记录的时间(dd, time)，T, H, P(Pa), if rain, W, ...
    """
    return mean_record


if __name__ == '__main__':
    print(retrieve_record(1))
