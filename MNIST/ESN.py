import random
import torch
from Win_file import reservoir
from tanh.active_func import ntanh,btanh,ptanh
device = torch.device('cuda:0')
#==============================================================================#

random.seed(25)
class ESN():
    def __init__(self, inSize, outSize, resSize, alpha, sparsity=0.3):
        self.inSize = inSize
        self.resSize = resSize
        self.outSize = outSize
        self.alpha = alpha
        self.sparsity = sparsity
        # +1 for Bias
        self.Win = (torch.randn(inSize+1, resSize, dtype=torch.float64,device=device))*0.15
        self.W = (torch.randint(size=(resSize, resSize), high=4, low=0,device=device))*1.0 # Reservoir --> Reservoir
        self.W[torch.randn(resSize, resSize,device=device) > self.sparsity] = 0




    def reservoir(self, data, new_start=True):
        self.dm = torch.zeros((data.shape[0], 1+self.resSize+self.inSize),device=device)
        if new_start:#
            self.R = (torch.randint(low=0,high=3,size=(1, self.resSize),device=device)-1)*0


        for t in range(data.shape[0]):
            self.R.to(device)
            u = torch.tensor(data[t], dtype=torch.float64,device=device)# initialize input at timestep t
            self.R = (1 - self.alpha)*self.R.cuda() +                              \
                self.alpha*ptanh(torch.matmul(torch.hstack((torch.tensor([1],dtype=torch.float64).to(device), u)), self.Win.to(device)) + \
                                   reservoir.poperation(self.R, self.W))
           self.dm[t] = torch.cat((torch.cat((torch.tensor([1],dtype=torch.float64,device=device),u), dim=0), self.R.flatten()),dim=0)

        return self.dm

