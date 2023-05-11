import torch


class Prediction(object):
    def __init__(self, TTPHW: torch.Tensor, description: str) -> None:
        assert isinstance(TTPHW, torch.Tensor)
        
        self.TTPHW = TTPHW  #! temp_min, temp_max, pressure, humidity, wind_speed
        self.description = description
        
        self.temp_min = TTPHW[0]
        self.temp_max = TTPHW[1]
        self.pressure = TTPHW[2]
        self.humidity = TTPHW[3]
        self.wind_speed = TTPHW[4]
    
    def __str__(self) -> str:
        ret = "Predictions:\n"
        
        ret += f"- Min Temperature: {self.temp_min} Degree C\n"
        ret += f"- Max Temperature: {self.temp_max} Degree C\n"
        ret += f"- Pressure: {self.pressure} hPa\n"
        ret += f"- Humidity: {self.humidity}%\n"
        ret += f"- Wind Speed: {self.wind_speed} m/s\n"
        ret += f"- Weather Description: {self.description}\n"
        ret += "\n"
    
        return ret
    