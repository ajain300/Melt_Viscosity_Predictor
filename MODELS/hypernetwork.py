import torch
import torch.nn as nn

class Visc_HyperNet(nn.Module):
    def __init__(self, n_fp, config = {"l1" : 120, "l2" : 120, "d1" : 0.2, "d2" : 0.2}):
        super(Visc_HyperNet, self).__init__()
        l1 = config["l1"]
        l2 = config["l2"]
        d1 = config["d1"]
        d2 = config["d2"]
        self.n_fp = n_fp
        self.layer_1 = nn.Linear(self.n_fp + 1, l1)
        self.rel = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.d1 = nn.Dropout(p = d1)
        if l2 > 0:    
            self.layer_2 = nn.Linear(l1, l2)
            self.d2 = nn.Dropout(p = d2)
            self.out = nn.Linear(l2, 11)
            self.layers = nn.ModuleList([self.layer_1, self.d1,self.rel,self.layer_2, self.d2,self.rel, self.out])
        else:
            self.out = nn.Linear(l1, 11)
            self.layers = nn.ModuleList([self.layer_1,self.d1, self.rel, self.out])

    def forward(self, fp, PDI):
        out = None

        x = torch.cat((fp, PDI), 1)
        x = x.view(-1,self.n_fp + 1)
        for l in self.layers:
            x = l(x)
        out = x

        #out[:,2:4] = self.tanh(out[:,2:4])
        #out[:,:2] = self.rel(out[:,:2])
        #out[:,4] = self.sig(out[:,4])
        #out[:,4:] = self.sig(out[:,4:])

        return out