import torch


class Prediction(object):
    def __init__(self, TPHW: torch.Tensor, description: str) -> None:
        self.TPHW = TPHW
        self.description = description
        
        self.temperature = TPHW[0]
        self.pressure = TPHW[1]
        self.humidity = TPHW[2]
        self.wind_speed = TPHW[3]
    
    def __str__(self) -> str:
        ret = ""
        
        ret += f"==> Temperature: {self.temperature}\n"
        ret += f"==> Pressure: {self.pressure}\n"
        ret += f"==> Humidity: {self.humidity}\n"
        ret += f"==> Wind Speed: {self.wind_speed}\n"
        ret += f"==> Weather Description: {self.description}\n"
    
        return ret
    