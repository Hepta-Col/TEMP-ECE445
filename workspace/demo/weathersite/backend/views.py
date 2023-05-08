from django.shortcuts import render, HttpResponse
from django.http import JsonResponse

import sqlite3
import time
import datetime

# Create your views here.


############### Data Retrieval #######################
def time_subtract(cur_time, timestamp):
    time_split = timestamp.split("_")
    # print(time_split)
    
    time_test = datetime.datetime(year = 2023, month = int(time_split[0].split(".")[0]), day = int(time_split[0].split(".")[1]),hour = int(time_split[1].split(".")[0]), minute = int(time_split[1].split(".")[1]), second = int(time_split[1].split(".")[2]))
    # print(time_test)
    timedelta = (cur_time-time_test).seconds
    return timedelta

def get_wind_level(speed):
    if speed <= 0.2:
        return 0, "Calm"
    elif speed <= 1.5:
        return 1, "Light air"
    elif speed <= 3.3:
        return 2, "Light breeze"
    elif speed <= 5.4:
        return 3, "Gentle breeze"
    elif speed <= 7.9:
        return 4, "Moderate breeze"
    elif speed <= 10.7:
        return 5, "Fresh breeze"
    elif speed <= 13.8:
        return 6, "Strong breeze"
    elif speed <= 17.1:
        return 7, "Near gale"
    elif speed <= 20.7:
        return 8, "Gale"
    elif speed <= 24.4:
        return 9, "Strong gale"
    elif speed <= 28.4:
        return 10, "Storm"
    elif speed <= 32.6:
        return 11, "Violent storm"

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

    wind_level, wind_msg = get_wind_level(wind/len(records))

    dict_ = {'day': latest_time.split("_")[0], 'time': latest_time.split("_")[1][:-3].replace(".",":"), 'temp':round(temp/len(records)), 'humi':round(humi/len(records)), 
            'pres':round(pres/(100*len(records))), 'pres_percentage':round((pres-86000)/(106000-86000), 2),'rain': if_rain, 'wind': round(wind/len(records), 1), 'wind_level': wind_level, 'wind_msg': wind_msg, 'uv':round(uv/len(records)),
            'pm10': round(pm1/len(records), 1), 'pm25': round(pm2_5/len(records), 1), 'pm100': round(pm10/len(records), 1)}
    return dict_

def retrieve_record(min):
    #time_check = time.localtime(time.time())
    time_check = datetime.datetime.now()
    print(time_check)

    conn = sqlite3.connect('C:\MyFiles\TEMP-ECE445\workspace\demo\weatherdata.db')
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
    #print(records)

# cursor = cur.execute("Select * from (select rownum no ,serv_id from serv_history_517 ) where no>10")

    print ("数据操作成功")
    conn.close()

    if len(records) == 0:
        return 
    
    mean_record = get_mean(records)
    print(mean_record)
    return mean_record

############### Data Retrieval #######################

# @action(methods=['get'], detail=False)
def get_data(request):
    data = retrieve_record(int(3))
    return JsonResponse(data)

def get_hourly_predict(request):
    data = {"7": [25,87,1003,2.7], "8": [26,77,1018,3.5]}
    return JsonResponse(data)

def get_daily_predict(request):
    data = {"Sat": [17,28],"Sun": [19, 25],"Mon": [17,26]}
    return JsonResponse(data)