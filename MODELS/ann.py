import torch
import torch.nn as nn

class Visc_ANN(nn.Module):
    def __init__(self, n_fp):
        '''
        :param input_dim: input feature dimension
        :param hidden_size: hidden dimension
        :param num_classes: total number of classes
        '''
        super(Visc_ANN, self).__init__()

        self.n_fp = n_fp
        self.layer_1 = nn.Linear(n_fp + 3, 120)
        self.rel = nn.Relu()
        self.layer_2 = nn.Linear(60, 1)
        self.layers = [self.layer_1, self.rel,self.layer_2, self.rel]


    def forward(self, fp, M, S, T):
        out = None

        x = x.view(-1,self.input_dim)
        for l in self.layers:
            x = l(x)
        out = x

        return out