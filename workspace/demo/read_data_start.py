import xlwt
import time
#import serial
from serial import Serial

#设置表格样式
def set_style(name,height,bold=False):
 style = xlwt.XFStyle()
 font = xlwt.Font()
 font.name = name
 font.bold = bold
 font.color_index = 4
 font.height = height
 style.font = font
 return style
#写Excel
def write_excel():
 if serial.isOpen():
  print ('串口已打开\n')
 f = xlwt.Workbook()
 sheet1 = f.add_sheet('arduino_data',cell_overwrite_ok=True)
 row0 = ["time","temp","humi","pres","if rain","wind","uv index","PM1.0","PM2.5","PM10"]
 time1=time.localtime(time.time())
 #写第一行
 for i in range(len(row0)):
  sheet1.write(0,i,row0[i],set_style('Times New Roman',220,True))
 i=1
 time.sleep(5)
 serial.flushInput()
 s = []
 while True:
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

    if len(s) == 10 and s[0] == "Start":
     try:
      if s[0] == "Start":
          s.pop(0)
      print("write final s:")
      print(s)
      s.insert(0,"{0}.{1}_{2:0>2d}.{3:0>2d}.{4:0>2d}".format(time_temp[1],time_temp[2],time_temp[3],time_temp[4],time_temp[5]))
      print(s)
      for j in range(len(s)):
       sheet1.write(i,j,s[j],set_style('Times New Roman',220,False))
      # serial.flushInput()     # 清空接收缓存区
      i = i+1
      s = []
      time.sleep(4)
     except ValueError:
      serial.flushInput()
      continue
  except KeyboardInterrupt:
   time2=time.localtime(time.time())
   f.save(r'/Users/xuanyuchen/Desktop/ZJUI/ece445/pySerial/{0}.{1}_{2:0>2d}.{3:0>2d}.{4:0>2d}-{5}.{6}_{7:0>2d}.{8:0>2d}.{9:0>2d}.xls'.format\
     (time1[1],time1[2],time1[3],time1[4],time1[5],
     time2[1],time2[2],time2[3],time2[4],time2[5]))
   serial.close()
   print(time1)
   print(time2)
   quit()
if __name__ == '__main__':
 # serial = Serial('/dev/cu.usbmodem1401',9600,timeout=0.5)
 serial = Serial('/dev/cu.usbmodem21101',9600,timeout=0.5)
 write_excel()