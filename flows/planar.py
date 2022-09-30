## implemented from https://github.com/mrsalehi/stupid-simple-norm-flow
import torch
import torch.nn as nn
from .NormalizingFlow import Flow
from nets.LTU import LTU

def newton_method(function, initial, iteration=100, convergence=torch.Tensor([0.0001, 0.0001]).to('cuda')):
            for i in range(iteration): 
                previous_data = initial.clone()
                value = function(initial)
                value.sum().backward()
                # update 
                initial.data -= (value / initial.grad).data
                # zero out current gradient to hold new gradients in next iteration 
                initial.grad.data.zero_() 
                # Check convergence. 
                # When difference current epoch result and previous one is less than 
                # convergence factor, return result.
                comp = torch.le(torch.abs(initial - previous_data).data, torch.tensor(convergence))
                
                if comp.all() == True:
                    return initial.data
            return initial.data # return our final after iteration

class Planar(nn.Module):
    """
    Planar flow as introduced in arXiv: 1505.05770
        f(z) = z + u * h(w * z + b)
    """
    
    def __init__(self, net):
        super().__init__()
        self.net = net
        
    def forward(self,x):
        
        # g = f^-1
        z = self.net(x)
            
        for name, param in self.net.named_parameters():
            if name == 'u' : 
                self.u = param
            elif name == 'w' : 
                self.w = param
            elif name == 'b' : 
                self.b = param
        
        affine = torch.mm(x, self.w.T) + self.b          # 2*1
        psi = (1 - nn.Tanh()(affine) ** 2) * self.w      # 2*2
        abs_det = (1 + torch.mm(self.u, psi.T)).abs()    # 1*2
        log_det = torch.log(1e-4 + abs_det).squeeze(0)   # 2
        
        return z, log_det
    
    def inverse(self, z):
        
        sol = torch.zeros(z.size()).to('cuda')
        for idx, sample in enumerate(z):
            sample.requires_grad_()
            s = newton_method(self.net, sample)
            sol[idx] = s
            
        sol = sol.reshape(z.size())    
        return sol
    
class PlanarFlow(Flow):   

    def __init__(self, net = LTU, dim=5, device='cuda'):
        Flow.__init__(self) 
        self.net = net
        self.dim = dim
        self.device = device
        self.bijections = []
        self.build()
        self.flow = nn.ModuleList(self.bijections)  

    def build(self): 
        for i in range(self.dim):
            self.bijections += [Planar(self.net)]
    
    
    
    
    