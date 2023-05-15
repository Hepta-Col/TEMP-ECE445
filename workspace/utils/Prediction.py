import torch


class Prediction(object):
    def __init__(self, TTPHW: torch.Tensor, description: str) -> None:
        assert isinstance(TTPHW, torch.Tensor)
        
        self.TTPHW = TTPHW  #! temp_min, temp_max, pressure, humidity, wind_speed
        self.description = description
        
        self.temp_min = round(TTPHW[0].item(), 2)
        self.temp_max = round(TTPHW[1].item(), 2)
        self.pressure = round(TTPHW[2].item(), 2)
        self.humidity = round(TTPHW[3].item(), 2)
        self.wind_speed = round(TTPHW[4].item(), 2)
    
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
    