from serial import Serial
import json
import time

from sqlalchemy import create_engine, MetaData
from sqlalchemy import Column, Integer, String, Table
from sqlalchemy.ext.declarative import declarative_base

import os

engine = create_engine('sqlite:///C:\MyFiles\TEMP-ECE445\data\weatherdata.db', echo=True)
Base = declarative_base()

class Weatherdata(Base):
    __tablename__ = 'weatherdata'

    id = Column(Integer, primary_key=True)
    time = Column(String(13))
    temperature = Column(String(5))
    humidity = Column(String(5))
    pressure = Column(String(6))
    rain = Column(String(1))
    wind = Column(String(4))
    uv = Column(String(2))
    pm1_0 = Column(String(2))
    pm2_5 = Column(String(2))
    pm10 = Column(String(2))

    def __str__(self):
        return self.id

Base.metadata.create_all(engine)

metadata = MetaData(engine)
weather_table = Table('weatherdata', metadata, autoload=True)
ins = weather_table.insert()


# serialPort = "/dev/cu.usbmodem1401"
serialPort = "COM3"
baudRate = 9600
timeout = 0.5
serial = Serial(serialPort, baudRate, timeout=timeout)

time.sleep(5)
serial.flushInput()
s = []
# 循环获取数据(条件始终为真)
while 1:
  try:
    time.sleep(.001)                    # delay of 1ms
    val = serial.readline()                # read complete line from serial output
    while not '\\n'in str(val):         # check if full data is received. 
        # This loop is entered only if serial read value doesn't contain \n
        # which indicates end of a sentence. 
        # str(val) - val is byte where string operation to check `\\n` 
        # can't be performed
        print("Reading")
        time.sleep(.001)                # delay of 1ms 
        temp = serial.readline()           # check for serial output.
        if not not temp.decode():       # if temp is not empty.
            val = (val.decode('utf-8')+temp.decode('utf-8')).encode('utf-8')
            # requrired to decode, sum, then encode because
            # long values might require multiple passes
    response = val.decode('utf-8')                  # decoding from bytes

    # size = serial.inWaiting()
    # if size != 0:
      # response = serial.read(size)  # 读取内容并显示
      # response = serial.readline()
      # print("Response:")
    print(response)
    #print(response.decode('utf-8').rstrip('\r\n').split("\t"))
    # s=response.decode('utf-8').rstrip('\r\n').split('\t')
    temp_list = response.rstrip('\r\n').split("\t")
    if len(temp_list) != 0 and temp_list[-1] == "":
      temp_list.pop(-1)
    s = s + temp_list
    # print(s)
    while len(s) != 0 and s[0] != "Start":
      s.pop(0)
    print(s)
    time_temp = time.localtime(time.time())

    # if len(s)< 10:
    #   serial.flushInput()
    #   continue
    # 读取接收到的数据的第一行
    # strData = ser.readline()
    # 把拿到的数据转为字符串(串口接收到的数据为bytes字符串类型,需要转码字符串类型)
    # strJson = str(strData, encoding='utf-8')
    if len(s) == 10 and s[0] == "Start":
      try:
        if s[0] == "Start":
          s.pop(0)
        print("write final s:")
        print(s)
        s.insert(0,"{0}.{1}_{2:0>2d}.{3:0>2d}.{4:0>2d}".format(time_temp[1],time_temp[2],time_temp[3],time_temp[4],time_temp[5]))
        # if len(s[6]) == 1:
        #    s[6] = "0" + s[6]
        print(s)

        ins = ins.values(time = s[0], temperature=s[1],
                             humidity = s[2], pressure = s[3], rain = s[4], wind = s[5], uv = s[6], pm1_0 = s[7], 
                             pm2_5 = s[8], pm10 = s[9])
        # 连接引擎
        conn = engine.connect()
        # 执行语句
        result = conn.execute(ins)

        # serial.flushInput()     # 清空接收缓存区
        s = []
        time.sleep(4)

      except ValueError:
        serial.flushInput()
        continue

  except KeyboardInterrupt:
    time2=time.localtime(time.time())
    print("当前接收到的数据为空")
    print(time2)
    quit()

