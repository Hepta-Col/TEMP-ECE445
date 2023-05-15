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
    
def AQI(pm25, pm100):
    # for pm25
    print(pm25, pm100)
    if pm25 <= 35:
        iaqi1 = ((50-0)/(35-0))*(pm25-0) + 0
    elif pm25 <= 75:
        iaqi1 = ((100-50)/(75-35))*(pm25-35) + 50
    elif pm25 <= 115:
        iaqi1 = ((150-100)/(115-75))*(pm25-75) + 100
    elif pm25 <= 150:
        iaqi1 = ((200-150)/(150-115))*(pm25-115) + 150
    elif pm25 <= 250:
        iaqi1 = ((300-200)/(250-150))*(pm25-150) + 200
    elif pm25 <= 350:
        iaqi1 = ((400-300)/(350-250))*(pm25-250) + 300
    elif pm25 <= 500:
        iaqi1 = ((500-400)/(500-350))*(pm25-350) + 400
    else:
        iaqi1 = 500

    # for pm100
    if pm100 <= 50:
        iaqi2 = ((50-0)/(50-0))*(pm100-0) + 0
    elif pm100 <= 150:
        iaqi2 = ((100-50)/(150-50))*(pm100-50) + 50
    elif pm100 <= 250:
        iaqi2 = ((150-100)/(250-150))*(pm100-150) + 100
    elif pm100 <= 350:
        iaqi2 = ((200-150)/(350-250))*(pm100-250) + 150
    elif pm100 <= 420:
        iaqi2 = ((300-200)/(420-350))*(pm100-350) + 200
    elif pm100 <= 500:
        iaqi2 = ((400-300)/(500-420))*(pm100-420) + 300
    elif pm100 <= 600:
        iaqi2 = ((500-400)/(600-500))*(pm100-500) + 400
    else:
        iaqi2 = 500

    aqi = max(iaqi1,iaqi2)

    if aqi <= 50:
        aqi_level = 1
    elif aqi <= 100:
        aqi_level = 2
    elif aqi <= 150:
        aqi_level = 3
    elif aqi <= 200:
        aqi_level = 4
    elif aqi <= 300:
        aqi_level = 5
    else:
        aqi_level = 6

    # print(int(iaqi1), int(iaqi2), int(aqi), aqi_level)

    return int(iaqi1), int(iaqi2), int(aqi), aqi_level

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

    iaqi1, iaqi2, aqi, aqi_level = AQI(pm2_5/len(records), pm10/len(records))

    dict_ = {'day': latest_time.split("_")[0], 'time': latest_time.split("_")[1][:].replace(".",":"), 'temp':round(temp/len(records)), 'humi':round(humi/len(records)), 
            'pres':round(pres/(100*len(records))), 'pres_percentage':round((pres-86000)/(106000-86000), 2),'rain': if_rain, 'wind': round(wind/len(records), 1), 'wind_level': wind_level, 'wind_msg': wind_msg, 'uv':round(uv/len(records)),
            'pm10': round(pm1/len(records), 1), 'pm25': round(pm2_5/len(records), 1), 'iaqi1':iaqi1, 'pm100': round(pm10/len(records), 1), 'iaqi2':iaqi2, 'aqi': aqi, 'aqi_level': aqi_level}
    return dict_
    

def retrieve_record(min):
    #time_check = time.localtime(time.time())
    time_check = datetime.datetime.now()
    print(time_check)

    conn = sqlite3.connect('C:\MyFiles\TEMP-ECE445\data\weatherdata.db')
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

    t5 = datetime.datetime(year = 2023, month = 5, day = 8, hour = 19, minute = 38, second = 10)
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

def get_hours(cur_hour):
    hour1 = (int(cur_hour) + 1 )%24
    hour2 = (hour1 + 1)%24
    hour3 = (hour2 + 1)%24
    hour4 = (hour3 + 1)%24
    return str(hour1), str(hour2), str(hour3), str(hour4)

def get_descript(des):
    # if des == 'few clouds':
    #     return 0
    # elif des == 'light rain':
    #     return 1
    # elif des == 'overcast clouds':
    #     return 2
    # elif des == 'sky is clear':
    #     return 3
    # elif des == 'light snow':
    #     return 4
    # elif des == 'broken clouds':
    #     return 5
    # elif des == 'scattered clouds':
    #     return 6
    # elif des == 'snow':
    #     return 7
    # elif des == 'moderate rain':
    #     return 8
    # elif des == 'heavy intensity rain':
    #     return 9
    # elif des == 'very heacy rain':
    #     return 10

    if des == 'sky is clear':
        return 0
    else:
        return 1

def get_hourly_predict(request):
    
    conn = sqlite3.connect('C:\MyFiles\TEMP-ECE445\data\prediction.db')
    cur = conn.cursor()
    print ("预测数据库打开成功")
    # print(cur.rowcount)   

    # cursor = cur.execute("SELECT max(rowid) from prediction")
    # max_line = cursor.fetchone()[0]
    # print(max_line)   
    cursor = cur.execute("SELECT * FROM prediction WHERE ROWID IN ( SELECT max( ROWID ) FROM prediction )")
    lastline = cursor.fetchall()[0]
    print(lastline)

    cur_time = lastline[1]
    hour1, hour2, hour3, hour4 = get_hours(cur_time.split("_")[1][:2])

    tmax1 = str(round(float(lastline[6])))
    tmax2 = str(round(float(lastline[7])))
    tmax3 = str(round(float(lastline[8])))
    tmax4 = str(round(float(lastline[9])))

    tmin1 = str(round(float(lastline[2])))
    tmin2 = str(round(float(lastline[3])))
    tmin3 = str(round(float(lastline[4])))
    tmin4 = str(round(float(lastline[5])))
    
    pres1 = round(float(lastline[14]))
    pres2 = round(float(lastline[15]))
    pres3 = round(float(lastline[16]))
    pres4 = round(float(lastline[17]))

    humi1 = round(float(lastline[10]))
    humi2 = round(float(lastline[11]))
    humi3 = round(float(lastline[12]))
    humi4 = round(float(lastline[13]))

    wind_1 = str(round(float(lastline[18]),1))
    wind_2 = str(round(float(lastline[19]),1))
    wind_3 = str(round(float(lastline[20]),1))
    wind_4 = str(round(float(lastline[21]),1))


    wind_level1, wind_msg1 = get_wind_level(float(wind_1))
    wind_level2, wind_msg2 = get_wind_level(float(wind_2))
    wind_level3, wind_msg3 = get_wind_level(float(wind_3))
    wind_level4, wind_msg4 = get_wind_level(float(wind_4))

    descript1 = get_descript(lastline[22])
    descript2 = get_descript(lastline[23])
    descript3 = get_descript(lastline[24])
    descript4 = get_descript(lastline[25])

    data = {"Hour1": hour1, "Hour2": hour2, "Hour3": hour3, "Hour4": hour4, "MaxT1": tmax1, "MinT1":tmin1, 
            "MaxT2": tmax2, "MinT2":tmin2, "MaxT3": tmax3, "MinT3":tmin3, "MaxT4": tmax4, "MinT4":tmin4, "Pres1": pres1, 
            "Pres2": pres2, "Pres3": pres3, "Pres4": pres4, "Wind1":wind_1, 'Wind_level1': wind_level1, 'Wind_msg1': wind_msg1,"Wind2":wind_2, 'Wind_level2': wind_level2, 'Wind_msg2': wind_msg2, "Wind3":wind_3, 
            'Wind_level3': wind_level3, 'Wind_msg3': wind_msg3, "Wind4":wind_4, 'Wind_level4': wind_level4, 'Wind_msg4': wind_msg4,
            "Humi1": humi1, "Humi2": humi2, "Humi3": humi3, "Humi4": humi4, "day": cur_time.split("_")[0], "time": cur_time.split("_")[1][:2]+"h",
            "des1": descript1, "des2": descript2, "des3": descript3, "des4": descript4
            }
    return JsonResponse(data)

def get_daily_predict(request):
    data = {"DAY1": ["TODAY",20,25],"DAY2": ["THU",17, 28],"DAY3": ["FRI",19,26],"max":28, "min":17, "day": "5.8", "time": "00:00",}
    return JsonResponse(data)