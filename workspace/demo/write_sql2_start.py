from serial import Serial
import json
import time

from sqlalchemy import create_engine, MetaData
from sqlalchemy import Column, Integer, String, Table
from sqlalchemy.ext.declarative import declarative_base

import os


def write_predictions(s):
    print("s: ", s)
    assert len(s) == 25
    
    engine = create_engine('sqlite:///C:\MyFiles\TEMP-ECE445\data\prediction.db', echo=True)
    Base = declarative_base()

    class Weatherdata(Base):
        __tablename__ = 'prediction'

        id = Column(Integer, primary_key=True)
        
        time = Column(String(13))
        
        temperature_min_1 = Column(String(5))
        temperature_min_2 = Column(String(5))
        temperature_min_3 = Column(String(5))
        temperature_min_4 = Column(String(5))
        
        temperature_max_1 = Column(String(5))
        temperature_max_2 = Column(String(5))
        temperature_max_3 = Column(String(5))
        temperature_max_4 = Column(String(5))
        
        humidity_1 = Column(String(5))
        humidity_2 = Column(String(5))
        humidity_3 = Column(String(5))
        humidity_4 = Column(String(5))
        
        pressure_1 = Column(String(6))
        pressure_2 = Column(String(6))
        pressure_3 = Column(String(6))
        pressure_4 = Column(String(6))
        
        wind_1 = Column(String(4))
        wind_2 = Column(String(4))
        wind_3 = Column(String(4))
        wind_4 = Column(String(4))
        
        description_1 = Column(String(3))
        description_2 = Column(String(3))
        description_3 = Column(String(3))
        description_4 = Column(String(3))
        
        def __str__(self):
            return self.id

    Base.metadata.create_all(engine)

    metadata = MetaData(engine)
    weather_table = Table('prediction', metadata, autoload=True)
    ins = weather_table.insert()

    ins = ins.values(
                     time = s[0], 
                     temperature_min_1 = s[1],
                     temperature_min_2 = s[2],
                     temperature_min_3 = s[3],
                     temperature_min_4 = s[4],
                     temperature_max_1 = s[5],
                     temperature_max_2 = s[6],
                     temperature_max_3 = s[7],
                     temperature_max_4 = s[8],
                     humidity_1 = s[9], 
                     humidity_2 = s[10], 
                     humidity_3 = s[11], 
                     humidity_4 = s[12], 
                     pressure_1 = s[13], 
                     pressure_2 = s[14], 
                     pressure_3 = s[15], 
                     pressure_4 = s[16], 
                     wind_1 = s[17], 
                     wind_2 = s[18], 
                     wind_3 = s[19], 
                     wind_4 = s[20], 
                     description_1 = s[21],
                     description_2 = s[22],
                     description_3 = s[23],
                     description_4 = s[24],
                    )

    # 连接引擎
    conn = engine.connect()
    
    # 执行语句
    result = conn.execute(ins)
