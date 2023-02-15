import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from .hypernetwork import Visc_HyperNet

class TargetNet(nn.Module):
    def __init__(self, n_fp, config,device):
        '''
        :param input_dim: input feature dimension
        :param hidden_size: hidden dimension
        :param num_classes: total number of classes
        '''
        super(TargetNet, self).__init__()

        self.device = device
        self.hyper = Visc_HyperNet(n_fp, config).to(self.device)
        self.sig = nn.Sigmoid()
        self.soft = nn.Softplus()
        self.tanh = nn.Tanh()
        self.Mw_layer = MolWeight(device)
        self.Shear_layer = ShearRate(device)


    def forward(self, fp, M, S, T, PDI, train = False, get_constants = False):
        eta = None
        
        M, S, T = torch.squeeze(M), torch.squeeze(S), torch.squeeze(T)
        params = self.hyper(fp, PDI)
        self.alpha_1 = self.sig(params[:,0])*torch.tensor(3).to(self.device)
        #print(f'alpha_1 {self.alpha_1.device}')
        self.alpha_2 = self.sig(params[:,1])*torch.tensor(6).to(self.device)
        self.k_cr = self.tanh(params[:,2])
        self.beta_M = torch.tensor(10).to(self.device) + self.sig(params[:,3])*torch.tensor(30).to(self.device)
        self.M_cr = self.tanh(params[:,4])
        self.C_1 = self.sig(params[:,5])*torch.tensor(2).to(self.device)
        self.C_2 = self.sig(params[:,6])*torch.tensor(2).to(self.device)
        self.T_r = self.tanh(params[:,7])
        self.n = self.sig(params[:,8])
        self.tau = self.tanh(params[:,9])
        self.beta_shear = self.sig(params[:,10])*torch.tensor(30).to(self.device)
        
        #Temp
        t_shift = T - self.T_r
        num = -self.C_1 * t_shift
        den = self.C_2 + t_shift
        a_t = num/den

        if (a_t > 2).any():
            print('caught invalid temp shift')
            filter_idx = (a_t > 2).nonzero(as_tuple=True)
            for i in filter_idx:
                print('C1', self.C_1[i], 'C2', self.C_2[i], 'T_r',self.T_r[i], 'T', T[i], 'T_shift', t_shift[i] )
        # print(f'a_t {a_t.device}')
        # print(f't_shift {t_shift.device}')


        #Mw
        #eta_01 = self.alpha_1*M + self.k_1
        #eta_02 = self.alpha_2*M + self.k_2
        #self.M_cr = (self.k_2-self.k_1)/(self.alpha_1 - self.alpha_2)
        #M_weight = (torch.tensor(1).to(self.device) + nn.Tanh()(torch.tensor(50).to(self.device)*(self.M_cr - M)))*torch.tensor(0.5).to(self.device)
        #eta_0 = M_weight*eta_01 + (torch.tensor(1).to(self.device) - M_weight)*eta_02 + a_t
        #B = self.b_1 * torch.tensor(50)
        #eta_0 = M + self.b_1 * torch.log(torch.tensor(1).to(self.device) + torch.exp(torch.tensor(50) * (M - self.M_cr))) + a_t
        # print('a2-a1')
        # print((self.alpha_2 - self.alpha_1))
        
        #NEW
        eta_0 = self.Mw_layer(M, self.alpha_1, self.alpha_2, self.beta_M, self.M_cr, self.k_cr + a_t)
        eta, S_sc = self.Shear_layer(S, eta_0, self.n, self.beta_shear, self.tau, a_t)

        eta = torch.unsqueeze(eta, -1)

        if torch.isnan(eta).any():
            print('invalid out')
            nan_idx = (torch.isnan(eta)==1).nonzero(as_tuple=True)
            for i in nan_idx:
                print('Mcr', self.M_cr[i],'a2-a1' ,(self.alpha_2 - self.alpha_1)[i], 'tau', self.tau[i],
                'S', S, 'M', M, 'T', T, 'S_sc', S_sc)

        if train:
            return eta, self.alpha_1, self.alpha_2
        elif get_constants:
            return eta, self.alpha_1, self.alpha_2, self.M_cr, self.k_cr + a_t, self.C_1, self.C_2, self.T_r, self.tau, self.n, eta_0
        else:
            return eta

class MolWeight(nn.Module):
    def __init__(self, device):
        super(MolWeight, self).__init__()
        self.device = device
    
    def forward(self, M, alpha_1, alpha_2, beta, Mcr, kcr):
        M_cent = M-Mcr
        M_sc = beta*M_cent
        f_M = alpha_1*M_cent + kcr + torch.pow(beta, -1)*(alpha_2 - alpha_1)*(torch.log(torch.tensor(1).to(self.device) + torch.exp(torch.tensor(-1).to(self.device)*M_sc)) + M_sc)
        if (f_M>2).any():
            print('invalid out')
            nan_idx = (torch.isnan(f_M)>1).nonzero(as_tuple=True)
            print('a1', alpha_1, 'a2',alpha_2 ,'bM', beta, 'Mcr', Mcr, 'kcr', kcr, 'M', M, 'F(M)', f_M)
        
        #print('a1', alpha_1, 'a2',alpha_2 ,'bM', beta, 'Mcr', Mcr, 'kcr', kcr)        
        
        return f_M

class ShearRate(nn.Module):
    def __init__(self, device):
        super(ShearRate, self).__init__()
        self.device = device
    
    def forward(self, S, eta_0, n, beta, tau, a_t):
        Scr = tau
        S_sc = beta*(S - a_t -Scr)
        f_S = eta_0 - torch.pow(beta, -1)*(n)*(torch.log(torch.tensor(1).to(self.device) + torch.exp(torch.tensor(-1).to(self.device)*S_sc)) + S_sc)
        if torch.isnan(f_S).any():
            print('invalid out')
            nan_idx = (torch.isnan(f_S)==1).nonzero(as_tuple=True)
            for i in nan_idx:
                print('eta0', eta_0[i], 'n',n[i], 'bshear', beta[i], 'tau', tau[i])
        if (eta_0>2).any():
            print('eta0', eta_0, 'n',n, 'bshear', beta, 'tau', tau)
        
        
        return f_S, S_sc
