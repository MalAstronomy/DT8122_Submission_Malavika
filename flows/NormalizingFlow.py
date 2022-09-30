import torch
import torch.nn as nn
from torch.distributions import Normal


class Flow(nn.Module):
    """
    Generic class for flow functions
    """

    def __init__(self, device = 'cuda'):
        super().__init__()
        self.device = device  
        
    @property    
    def base_dist(self):
        return Normal(
            loc=torch.zeros(2, device=self.device),
            scale=torch.ones(2, device=self.device), validate_args=False
        )
              
    def build(self): 
        return NotImplemented
        
    def flow_outputs(self, z):
        log_det = torch.zeros(z.shape[0], device=self.device)
       
        for bijection in self.flow:
            z, ldj = bijection(z)
            log_det += ldj      
        return z, log_det
    
    def sample(self, num_samples):
        z = self.base_dist.sample((num_samples,))
        for bijection in reversed(self.flow):
            z = bijection.inverse(z)
        return z
    

    
    

   

           
        
    
    