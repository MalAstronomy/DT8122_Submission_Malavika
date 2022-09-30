import torch
from torch import nn

class MLP(nn.Module):
    """
    A multilayer perceptron with GELU nonlinearities
    """
    def __init__(self, ip = 1, out = 2):
        super().__init__()
        
        self.net = nn.Sequential(nn.Linear(ip,256), nn.GELU(),
                       nn.Linear(256,128), nn.GELU(),
                       nn.Linear(128,out))
        
    def forward(self, x):
        z = self.net(x)
        
        return z

