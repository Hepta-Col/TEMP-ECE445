import sqlite3
import time
import datetime

from statistics import mean 


def time_subtract(cur_time, timestamp):
    time_split = timestamp.split("_")
    # print(time_split)
    
    time_test = datetime.datetime(year = 2023, month = int(time_split[0].split(".")[0]), day = int(time_split[0].split(".")[1]),hour = int(time_split[1].split(".")[0]), minute = int(time_split[1].split(".")[1]), second = int(time_split[1].split(".")[2]))
    # print(time_test)
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
    """fetch newest min data"""
    #time_check = time.localtime(time.time())
    time_check = datetime.datetime.now()
    print(time_check)

    conn = sqlite3.connect('weatherdata.db')
    cur = conn.cursor()
    print ("数据库打开成功")
    # print(cur.rowcount)   

    cursor = cur.execute("SELECT max(rowid) from weatherdata")
    max_line = cursor.fetchone()[0]
    print(max_line)   

    query = """SELECT * FROM weatherdata WHERE rowid > ?"""
    cursor = cur.execute(query, (max_line-15*min,))
    records = cursor.fetchall()
    print(records)
    print("\n")
    # print(record[0][1])
    # while 1:
    #     if records[0][1]:
    #         print(records[0][1])
    cur.close()   

    t5 = datetime.datetime(year = 2023, month = 5, day = 7, hour = 20, minute = 22, second = 10)
    while 1:
        if len(records)!= 0 and time_subtract(time_check, records[0][1]) <= 60*min:
        # if len(records)!= 0 and time_subtract(t5, records[0][1]) <= 60*min:
            break
        else:
            if len(records) != 0:
                records.pop(0)
            else:
                break
    print(records)

# cursor = cur.execute("Select * from (select rownum no ,serv_id from serv_history_517 ) where no>10")

    print ("数据操作成功")
    conn.close()

    if len(records) == 0:
        return 
    
    mean_record = get_mean(records)
    """格式
    1286, '5.8_20.34.02', '24.10', '56.00', '101863', '0', '4.11', '0', '18', '25', '26'
    行数(id)，记录的时间(dd, time)，T, H, P(Pa), if rain, W, ...
    """
    print(mean_record)



retrieve_record(int(1))
# time.sleep(30)
# retrieve_record(int(1))
