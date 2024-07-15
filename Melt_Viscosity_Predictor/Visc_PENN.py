# This is the architecture for the Physics Enforced Neural Network for polymer melt viscosity.
# It contains 4 classes: Visc_PENN for the overall architecture, MLP, for the MLP part of PENN
# MolWeight and ShearRate classes which encode eta-Mw and eta-shear_rate trends.
# Author: ayush.jain@gatech.edu 
#

import torch
import torch.nn as nn
from enum import Enum
from utils.train_torch import LossTypes
from utils.train_utils import EarlyStopping
from utils.gradnorm import GradNorm
import os
from torchviz import make_dot
import wandb

['a1', 'a2', 'M_cr', 'k_1', 'c1', 'c2', 'Tr','tau', 'n', 'eta_0']
class Visc_Constants(Enum):
    a1 = 'alpha_1'
    a2 = 'alpha_2'
    k1 = 'k_1'
    k2 = 'k_2'
    kcr = 'k_cr'
    Mcr = 'M_cr'
    c1 = 'C1'
    c2 = 'C2'
    Tr = 'T_r'
    Scr = 'S_cr'
    n = 'n'
    beta_M = 'Beta_Mw'
    beta_shear = 'Beta_Shear'
    lnA = 'lnA'
    EaR = 'EaR'
class Visc_PENN_Base(nn.Module):
    def __init__(self, n_fp, config,device, apply_gradnorm = False, n_params = 11, **kwargs):
        '''
        :param n_fp (int): input fingerprint dimension
        :param config (dict): configurations for hyperparameters for the MLP
        :param device: device, either cuda:X or cpu
        '''
        super(Visc_PENN_Base, self).__init__()

        self.training_complete = False

        self.config = config
        self.device = device
        self.mlp = MLP_PENN(n_fp, config = config, latent_param_size= n_params).to(self.device)
        self.rel = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.soft = nn.Softplus()
        self.tanh = nn.Tanh()
        self.Mw_layer = MolWeight(device)
        self.Shear_layer = ShearRate(device)
        self.losses = []
        self.fold = kwargs.get("fold", None)
        self.run =kwargs.get("run", None)
        self.early_stopping = EarlyStopping()

        for param in self.mlp.parameters():
            param.requires_grad = True
        
        if apply_gradnorm:
            print("Initialize gradnorm.")
            self.grad_norm = GradNorm(layer=self.mlp.out, alpha = 2.5,lr2 = 1.0,device = device, log= True)
            self.mlp.out.register_full_backward_hook(self.grad_norm.gradNorm_layer)
        else:
            self.grad_norm = None

    def train_model(self, dataloader, optimizer, criterion, config, scheduler = None, **kwargs):
        self.train()
        # Record total loss
        final_loss_dict = {}
        # Get the progress bar for later modification
        # Mini-batch training

        for batch_idx, (XX, M, S, T, P, visc) in enumerate(dataloader):

            optimizer.zero_grad()
            
            step_losses = self.train_step(XX, M, S, T, P, visc, criterion, optimizer)

            if self.run:
                # Log the training losses into WandB
                if self.grad_norm:
                    self.run.log({**{f"{k}_fold{self.fold}": v for k, v in step_losses.items()},
                                    **{f"{k}_fold{self.fold}" : v for k,v in self.grad_norm.get_current_weights().items()}})
                else:
                    self.run.log({f"{k}_fold{self.fold}": v for k, v in step_losses.items()})
            
            optimizer.step()

            for key, val in step_losses.items():
                if key in final_loss_dict.keys():
                    final_loss_dict[key] += val
                else:
                    final_loss_dict[key] = val
                
        for key, val in step_losses.items():
            if key != LossTypes.tot_train.value:
                final_loss_dict[key] /= len(dataloader)
        final_loss_dict[LossTypes.avg_train.value] = final_loss_dict[LossTypes.tot_train.value] / len(dataloader)
        
        self.losses.append(final_loss_dict)

        return final_loss_dict

    def train_step(self, XX, M, S, T, P, visc, criterion, optimizer) -> dict:
        out, a1,a2 = self.forward(XX, M, S, T, P, train = True)
        loss = criterion(out, visc)
        a1_loss = self.config["a_weight"]*criterion(a1, torch.ones_like(a1).to(self.device))
        a2_loss = self.config["a_weight"]*criterion(a2, torch.ones_like(a2)*torch.tensor(3.4).to(self.device)) 
        loss = loss + a2_loss + a1_loss

        loss.backward()
        # if self.grad_norm is not None:
        #     for name, param in self.named_parameters():
        #         if param.requires_grad:
        #             print(name, param.grad)
        #     loss = self.grad_norm.gradNorm_layer(loss, optimizer, lr2 = 0.0001)
        
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        return {LossTypes.tot_train.value : loss, LossTypes.a1.value : a1_loss, LossTypes.a2.value : a2_loss}

    def save_checkpoint(self, save_path, optimizer, scheduler, epoch, complete : bool = False):
        """
        Saves a checkpoint of the model and training state.
        
        Parameters:
        - save_path: Path to save the checkpoint.
        - model: The PyTorch model to save.
        - optimizer: The optimizer being used.
        - epoch: The current epoch number.
        - loss: The loss at the checkpoint.
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'losses': self.losses,
            'training_complete': complete
        }
        torch.save(checkpoint, save_path)

        # if self.grad_norm is not None:
        #     dir_path = os.path.dirname(save_path)
        #     self.grad_norm.plot_gradnorm_stats(os.path.join(dir_path, "gradnorm.png"))
        #     print(f"SAVED GRADNORM at {os.path.join(dir_path, 'gradnorm.png')}")

    def load_checkpoint(self, load_path, optimizer, scheduler):
        """
        Loads a checkpoint into the model and optimizer.
        
        Parameters:
        - load_path: Path to the checkpoint to be loaded.
        - model: The PyTorch model to load the state into.
        - optimizer: The optimizer to load the state into.
        """
        checkpoint = torch.load(load_path)
        self.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        self.losses = checkpoint['losses']

        if 'training_complete' in checkpoint.keys():
            self.training_complete = checkpoint['training_complete']
        else:
            self.training_complete = True

        return epoch, optimizer, scheduler 

    def load_trained_model(self, load_path) -> None:
        checkpoint = torch.load(load_path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.losses = checkpoint['losses']

    def evaluate(self, dataloader, criterion):
        # Set the model to eval mode to avoid weights update
        self.eval()
        total_loss = 0.
        with torch.no_grad():
            # Get the progress bar 
            for batch_idx, (XX, M, S, T, P, visc) in enumerate(dataloader):
            
                with torch.no_grad():
                    out = self.forward(XX, M, S, T, P)

                loss = criterion(out, visc)

                total_loss += loss
        
        avg_loss = total_loss / len(dataloader)

        # append the validation loss information to the list of loss dictionaries of the current (therefore latest epoch)
        self.losses[-1].update({
            LossTypes.tot_val.value : total_loss,
            LossTypes.avg_val.value : avg_loss
        })
        return total_loss, avg_loss

class Visc_ANN(Visc_PENN_Base):
    # Inherits the Visc_PENN_Base class to have training fuctions the same at a high level 
    # but we want to change it.
    def __init__(self, n_fp, config, device, **kwargs):
        super(Visc_PENN_Base, self).__init__()
        # initialize hyperparameters
        l1 = config["l1"]
        l2 = config["l2"]
        d1 = config["d1"]
        d2 = config["d2"]
        self.n_fp = n_fp
        self.layer_1 = nn.Linear(self.n_fp + 4, l1)
        self.d1 = nn.Dropout(p = d1)
        self.rel = nn.ReLU()
        if l2 > 0:    
            self.layer_2 = nn.Linear(l1, l2)
            self.d2 = nn.Dropout(p = d2)
            self.out = nn.Linear(l2, 1)
            self.layers = nn.ModuleList([self.layer_1, self.d1,self.rel,self.layer_2, self.d2,self.rel, self.out])
        else:
            self.out = nn.Linear(l1, 1)
            self.layers = nn.ModuleList([self.layer_1,self.d1, self.rel, self.out])

        self.run =kwargs.get("run", None)
        self.early_stopping = EarlyStopping()
        self.losses = []
        self.fold = kwargs.get("fold", None)
        self.device = device
        self.training_complete = False
        self.grad_norm = None 

    def forward(self, fp, Mw, Shear, T, PDI):
        out = None

        x = torch.cat((fp, Mw, Shear, T, PDI), 1)
        x = x.view(-1,self.n_fp + 4)
        for l in self.layers:
            x = l(x)
        out = x

        return out

    def train_step(self, XX, M, S, T, P, visc, criterion, optimizer) -> dict:
        out = self.forward(XX, M, S, T, P)
        loss = criterion(out, visc)

        loss.backward()
        # if self.grad_norm is not None:
        #     for name, param in self.named_parameters():
        #         if param.requires_grad:
        #             print(name, param.grad)
        #     loss = self.grad_norm.gradNorm_layer(loss, optimizer, lr2 = 0.0001)
        
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        return {LossTypes.tot_train.value : loss}

class Visc_PENN_WLF(Visc_PENN_Base):
    def forward(self, fp, M, S, T, PDI, train:bool = False, get_constants:bool = False):
        """
        Performs a forward pass through the PENN
        
        :param fp: The fingerprint vector of size n_fp
        :param M: The scaled log mol. weight of the polymer melt
        :param S: The scaled shear rate of the polymer melt 
        :param T: The scaled temperature of the polymer melt
        :param PDI: The scaled polydispersity index of the polymer melt
        :param train (bool): If its set to true
        A boolean parameter that indicates whether the forward pass is being performed during training or not. 
        If set to True, it returns all quantities that are needed for the loss calculation. In this case its the calcualted eta, alpha_1, and alpha_2
        :param get_constants (bool): Boolean flag that indicates whether or not to return the empirical constants calculated by the
        forward pass of the MLP. defaults to False (optional)
        """
        eta = None
        
        M, S, T = torch.squeeze(M), torch.squeeze(S), torch.squeeze(T)
        params = self.mlp(fp, PDI)
        self.alpha_1 = self.sig(params[:,0])*torch.tensor(3).to(self.device)
        self.alpha_2 = self.sig(params[:,1])*torch.tensor(6).to(self.device)
        self.k_1 = torch.tensor(2).to(self.device) * self.tanh(params[:,2]) - torch.tensor(1.0).to(self.device)
        self.beta_M = torch.tensor(20).to(self.device) + self.sig(params[:,3])*torch.tensor(40).to(self.device)
        self.M_cr = self.tanh(params[:,4])
        self.C_1 = self.sig(params[:,5])*torch.tensor(2).to(self.device)
        self.C_2 = self.sig(params[:,6])*torch.tensor(2).to(self.device)
        self.T_r = self.tanh(params[:,7]) - 1.0
        self.n = self.sig(params[:,8])
        self.crit_shear = self.tanh(params[:,9])
        self.beta_shear = torch.tensor(10).to(self.device) + self.sig(params[:,10])*torch.tensor(30).to(self.device)
        
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
        
        #NEW
        eta_0 = self.Mw_layer(M, self.alpha_1, self.alpha_2, self.beta_M, self.M_cr, self.k_1 + a_t)
        eta = self.Shear_layer(S, eta_0, self.n, self.beta_shear, self.crit_shear)

        eta = torch.unsqueeze(eta, -1)

        if torch.isnan(eta).any():
            print('invalid out')
            nan_idx = (torch.isnan(eta)==1).nonzero(as_tuple=True)
            for i in nan_idx:
                print('Mcr', self.M_cr[i],'a2-a1' ,(self.alpha_2 - self.alpha_1)[i], 'crit_shear', self.crit_shear[i],
                'S', S, 'M', M, 'T', T)

        if train:
            return eta, self.alpha_1, self.alpha_2
        elif get_constants:
            return {Visc_Constants.a1.value : self.alpha_1, 
                    Visc_Constants.a2.value :    self.alpha_2, 
                    Visc_Constants.Mcr.value :    self.M_cr, 
                    Visc_Constants.k1.value :    self.k_1, 
                    Visc_Constants.c1.value :    self.C_1, 
                    Visc_Constants.c2.value :    self.C_2, 
                    Visc_Constants.Tr.value :    self.T_r,
                    Visc_Constants.Scr.value :    self.crit_shear, 
                    Visc_Constants.n.value :    self.n,
                    Visc_Constants.beta_M.value :    self.beta_M,
                    Visc_Constants.beta_shear.value :    self.beta_shear}
        else:
            return eta

class Visc_PENN_WLF_Hybrid(Visc_PENN_Base):
    def __init__(self, n_fp, config,device, apply_gradnorm = False, n_params = 11, **kwargs):
        '''
        :param n_fp (int): input fingerprint dimension
        :param config (dict): configurations for hyperparameters for the MLP
        :param device: device, either cuda:X or cpu
        '''
        super(Visc_PENN_Base, self).__init__()

        self.training_complete = False

        self.config = config
        self.device = device
        self.mlp = MLP_PENN(n_fp, config = config, latent_param_size= n_params).to(self.device)
        self.sig = nn.Sigmoid()
        self.soft = nn.Softplus()
        self.tanh = nn.Tanh()
        self.Mw_layer = MolWeight(device)
        self.Shear_layer = ShearRate_softplus(device)
        self.losses = []
        self.fold = kwargs.get("fold", None)
        self.run =kwargs.get("run", None)
        self.early_stopping = EarlyStopping()

        for param in self.mlp.parameters():
            param.requires_grad = True
        
        if apply_gradnorm:
            print("Initialize gradnorm.")
            self.grad_norm = GradNorm(layer=self.mlp.out, alpha = 2.5,lr2 = 1.0,device = device, log= True)
            self.mlp.out.register_full_backward_hook(self.grad_norm.gradNorm_layer)
        else:
            self.grad_norm = None
    
    def forward(self, fp, M, S, T, PDI, train:bool = False, get_constants:bool = False):
        """
        Performs a forward pass through the PENN
        
        :param fp: The fingerprint vector of size n_fp
        :param M: The scaled log mol. weight of the polymer melt
        :param S: The scaled shear rate of the polymer melt 
        :param T: The scaled temperature of the polymer melt
        :param PDI: The scaled polydispersity index of the polymer melt
        :param train (bool): If its set to true
        A boolean parameter that indicates whether the forward pass is being performed during training or not. 
        If set to True, it returns all quantities that are needed for the loss calculation. In this case its the calcualted eta, alpha_1, and alpha_2
        :param get_constants (bool): Boolean flag that indicates whether or not to return the empirical constants calculated by the
        forward pass of the MLP. defaults to False (optional)
        """
        eta = None
        
        M, S, T = torch.squeeze(M), torch.squeeze(S), torch.squeeze(T)
        params = self.mlp(fp, PDI)
        self.alpha_1 = self.sig(params[:,0])*torch.tensor(3).to(self.device)
        self.alpha_2 = self.sig(params[:,1])*torch.tensor(6).to(self.device)
        self.k_1 = torch.tensor(2).to(self.device) * self.tanh(params[:,2]) - torch.tensor(1.0).to(self.device)
        self.beta_M = torch.tensor(20).to(self.device) + self.sig(params[:,3])*torch.tensor(40).to(self.device)
        self.M_cr = self.tanh(params[:,4])
        self.C_1 = self.sig(params[:,5])*torch.tensor(2).to(self.device)
        self.C_2 = self.sig(params[:,6])*torch.tensor(2).to(self.device)
        self.T_r = self.tanh(params[:,7]) - 1.0
        self.n = self.sig(params[:,8])
        self.crit_shear = self.tanh(params[:,9])
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
        
        #NEW
        eta_0 = self.Mw_layer(M, self.alpha_1, self.alpha_2, self.beta_M, self.M_cr, self.k_1 + a_t)
        eta = self.Shear_layer(S, eta_0, self.n, self.beta_shear, self.crit_shear)

        eta = torch.unsqueeze(eta, -1)

        if torch.isnan(eta).any():
            print('invalid out')
            nan_idx = (torch.isnan(eta)==1).nonzero(as_tuple=True)
            for i in nan_idx:
                print('Mcr', self.M_cr[i],'a2-a1' ,(self.alpha_2 - self.alpha_1)[i], 'crit_shear', self.crit_shear[i],
                'S', S, 'M', M, 'T', T)

        if train:
            return eta, self.alpha_1, self.alpha_2
        elif get_constants:
            return {Visc_Constants.a1.value : self.alpha_1, 
                    Visc_Constants.a2.value :    self.alpha_2, 
                    Visc_Constants.Mcr.value :    self.M_cr, 
                    Visc_Constants.k1.value :    self.k_1, 
                    Visc_Constants.c1.value :    self.C_1, 
                    Visc_Constants.c2.value :    self.C_2, 
                    Visc_Constants.Tr.value :    self.T_r,
                    Visc_Constants.Scr.value :    self.crit_shear, 
                    Visc_Constants.n.value :    self.n,
                    Visc_Constants.beta_M.value :    self.beta_M,
                    Visc_Constants.beta_shear.value :    self.beta_shear}
        else:
            return eta

class Visc_PENN_WLF_SP(Visc_PENN_Base):
    def __init__(self, n_fp, config,device, apply_gradnorm = False, n_params = 11, **kwargs):
        '''
        :param n_fp (int): input fingerprint dimension
        :param config (dict): configurations for hyperparameters for the MLP
        :param device: device, either cuda:X or cpu
        '''
        super(Visc_PENN_Base, self).__init__()

        self.config = config
        self.device = device
        self.mlp = MLP_PENN(n_fp, config = config, latent_param_size= n_params).to(self.device)
        self.sig = nn.Sigmoid()
        self.soft = nn.Softplus()
        self.tanh = nn.Tanh()
        self.Mw_layer = MolWeight_softplus(device)
        self.Shear_layer = ShearRate_softplus(device)
        self.losses = []
        self.fold = kwargs.get("fold", None)
        self.run =kwargs.get("run", None)
        self.early_stopping = EarlyStopping()
        self.training_complete = False

        for param in self.mlp.parameters():
            param.requires_grad = True
        
        if apply_gradnorm:
            print("Initialize gradnorm.")
            self.grad_norm = GradNorm(layer=self.mlp.out, alpha = 2.5,lr2 = 10.0,device = device, log= True)
            self.mlp.out.register_full_backward_hook(self.grad_norm.gradNorm_layer)
        else:
            self.grad_norm = None
    
    def forward(self, fp, M, S, T, PDI, train:bool = False, get_constants:bool = False):
        """
        Performs a forward pass through the PENN
        
        :param fp: The fingerprint vector of size n_fp
        :param M: The scaled log mol. weight of the polymer melt
        :param S: The scaled shear rate of the polymer melt 
        :param T: The scaled temperature of the polymer melt
        :param PDI: The scaled polydispersity index of the polymer melt
        :param train (bool): If its set to true
        A boolean parameter that indicates whether the forward pass is being performed during training or not. 
        If set to True, it returns all quantities that are needed for the loss calculation. In this case its the calcualted eta, alpha_1, and alpha_2
        :param get_constants (bool): Boolean flag that indicates whether or not to return the empirical constants calculated by the
        forward pass of the MLP. defaults to False (optional)
        """
        eta = None
        
        M, S, T = torch.squeeze(M), torch.squeeze(S), torch.squeeze(T)
        
        params = self.mlp(fp, PDI)
        self.alpha_1 = self.sig(params[:,0])*torch.tensor(3).to(self.device)
        self.alpha_2 = self.sig(params[:,1])*torch.tensor(6).to(self.device)
        self.k_cr = self.tanh(params[:,2])
        self.beta_M = torch.tensor(10).to(self.device) + self.sig(params[:,3])*torch.tensor(30).to(self.device)
        self.M_cr = self.tanh(params[:,4])
        self.C_1 = self.sig(params[:,5])*torch.tensor(2).to(self.device)
        self.C_2 = self.sig(params[:,6])*torch.tensor(2).to(self.device)
        self.T_r = self.tanh(params[:,7]) - 1.0
        self.n = self.sig(params[:,8])
        self.crit_shear = self.tanh(params[:,9])
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
        
        #NEW
        eta_0 = self.Mw_layer(M, self.alpha_1, self.alpha_2, self.beta_M, self.M_cr, self.k_cr + a_t)
        eta = self.Shear_layer(S, eta_0, self.n, self.beta_shear, self.crit_shear)

        eta = torch.unsqueeze(eta, -1)

        if torch.isnan(eta).any():
            print('invalid out')
            nan_idx = (torch.isnan(eta)==1).nonzero(as_tuple=True)
            for i in nan_idx:
                print('Mcr', self.M_cr[i],'a2-a1' ,(self.alpha_2 - self.alpha_1)[i], 'crit_shear', self.crit_shear[i],
                'S', S, 'M', M, 'T', T, 'S_sc', S_sc)

        if train:
            return eta, self.alpha_1, self.alpha_2
        elif get_constants:
            return {Visc_Constants.a1.value : self.alpha_1, 
                    Visc_Constants.a2.value :    self.alpha_2, 
                    Visc_Constants.Mcr.value :    self.M_cr, 
                    Visc_Constants.kcr.value :    self.k_cr, 
                    Visc_Constants.c1.value :    self.C_1, 
                    Visc_Constants.c2.value :    self.C_2, 
                    Visc_Constants.Tr.value :    self.T_r,
                    Visc_Constants.Scr.value :    self.crit_shear, 
                    Visc_Constants.n.value :    self.n,
                    Visc_Constants.beta_M.value :    self.beta_M,
                    Visc_Constants.beta_shear.value :    self.beta_shear}
        else:
            return eta

class Visc_PENN_Arrhenius_SP(Visc_PENN_Base):
    def __init__(self, n_fp, config,device, apply_gradnorm = False, n_params = 10, **kwargs):
        '''
        :param n_fp (int): input fingerprint dimension
        :param config (dict): configurations for hyperparameters for the MLP
        :param device: device, either cuda:X or cpu
        '''
        super(Visc_PENN_Base, self).__init__()

        self.config = config
        self.device = device
        self.mlp = MLP_PENN(n_fp, config = config, latent_param_size= n_params).to(self.device)
        self.sig = nn.Sigmoid()
        self.soft = nn.Softplus()
        self.tanh = nn.Tanh()
        self.Mw_layer = MolWeight_softplus(device)
        self.Shear_layer = ShearRate_softplus(device)
        self.losses = []
        self.fold = kwargs.get("fold", None)
        self.run =kwargs.get("run", None)
        self.early_stopping = EarlyStopping()
        self.training_complete = False

        for param in self.mlp.parameters():
            param.requires_grad = True
        
        if apply_gradnorm:
            print("Initialize gradnorm.")
            self.grad_norm = GradNorm(layer=self.mlp.out, alpha = 2.5,lr2 = 10.0,device = device, log= True)
            self.mlp.out.register_full_backward_hook(self.grad_norm.gradNorm_layer)
        else:
            self.grad_norm = None
    
    def forward(self, fp, M, S, T, PDI, train:bool = False, get_constants:bool = False):
        """
        Performs a forward pass through the PENN
        
        :param fp: The fingerprint vector of size n_fp
        :param M: The scaled log mol. weight of the polymer melt
        :param S: The scaled shear rate of the polymer melt 
        :param T: The scaled temperature of the polymer melt
        :param PDI: The scaled polydispersity index of the polymer melt
        :param train (bool): If its set to true
        A boolean parameter that indicates whether the forward pass is being performed during training or not. 
        If set to True, it returns all quantities that are needed for the loss calculation. In this case its the calcualted eta, alpha_1, and alpha_2
        :param get_constants (bool): Boolean flag that indicates whether or not to return the empirical constants calculated by the
        forward pass of the MLP. defaults to False (optional)
        """
        eta = None
        
        M, S, T = torch.squeeze(M), torch.squeeze(S), torch.squeeze(T)
        
        params = self.mlp(fp, PDI)
        self.alpha_1 = self.sig(params[:,0])*torch.tensor(3).to(self.device)
        self.alpha_2 = self.sig(params[:,1])*torch.tensor(6).to(self.device)
        self.k_cr = self.tanh(params[:,2])
        self.beta_M = torch.tensor(10).to(self.device) + self.sig(params[:,3])*torch.tensor(30).to(self.device)
        self.M_cr = self.tanh(params[:,4])
        self.lnA = params[:,5]
        self.EaR = params[:,6]
        self.n = self.sig(params[:,7])
        self.crit_shear = self.tanh(params[:,8])
        self.beta_shear = self.sig(params[:,9])*torch.tensor(30).to(self.device)

        #Temp
        a_t = self.lnA + self.EaR*T

        if (a_t > 2).any():
            print('caught invalid temp shift')
            filter_idx = (a_t > 2).nonzero(as_tuple=True)
            for i in filter_idx:
                print('C1', self.C_1[i], 'C2', self.C_2[i], 'T_r',self.T_r[i], 'T', T[i], 'T_shift', t_shift[i] )
        
        #NEW
        eta_0 = self.Mw_layer(M, self.alpha_1, self.alpha_2, self.beta_M, self.M_cr, self.k_cr + a_t)
        eta = self.Shear_layer(S, eta_0, self.n, self.beta_shear, self.crit_shear)

        eta = torch.unsqueeze(eta, -1)

        if torch.isnan(eta).any():
            print('invalid out')
            nan_idx = (torch.isnan(eta)==1).nonzero(as_tuple=True)
            for i in nan_idx:
                print('Mcr', self.M_cr[i],'a2-a1' ,(self.alpha_2 - self.alpha_1)[i], 'crit_shear', self.crit_shear[i],
                'S', S, 'M', M, 'T', T, 'S_sc', S_sc)

        if train:
            return eta, self.alpha_1, self.alpha_2
        elif get_constants:
            return {Visc_Constants.a1.value : self.alpha_1, 
                    Visc_Constants.a2.value :    self.alpha_2, 
                    Visc_Constants.Mcr.value :    self.M_cr, 
                    Visc_Constants.kcr.value :    self.k_cr, 
                    Visc_Constants.lnA.value :    self.lnA, 
                    Visc_Constants.EaR.value :    self.EaR, 
                    Visc_Constants.Scr.value :    self.crit_shear, 
                    Visc_Constants.n.value :    self.n,
                    Visc_Constants.beta_M.value :    self.beta_M,
                    Visc_Constants.beta_shear.value :    self.beta_shear}
        else:
            return eta

class Visc_PENN_Arrhenius(Visc_PENN_Base):
    def forward(self, fp, M, S, T, PDI, train:bool = False, get_constants:bool = False):
        """
        Performs a forward pass through the PENN
        
        :param fp: The fingerprint vector of size n_fp
        :param M: The scaled log mol. weight of the polymer melt
        :param S: The scaled shear rate of the polymer melt 
        :param T: The scaled temperature of the polymer melt - MAKE SURE THAT T = 1/T
        :param PDI: The scaled polydispersity index of the polymer melt
        :param train (bool): If its set to true
        A boolean parameter that indicates whether the forward pass is being performed during training or not. 
        If set to True, it returns all quantities that are needed for the loss calculation. In this case its the calcualted eta, alpha_1, and alpha_2
        :param get_constants (bool): Boolean flag that indicates whether or not to return the empirical constants calculated by the
        forward pass of the MLP. defaults to False (optional)
        """
        eta = None
        
        M, S, T = torch.squeeze(M), torch.squeeze(S), torch.squeeze(T)
        params = self.mlp(fp, PDI)
        self.alpha_1 = self.sig(params[:,0])*torch.tensor(3).to(self.device)
        self.alpha_2 = self.sig(params[:,1])*torch.tensor(6).to(self.device)
        self.k_1 = self.tanh(params[:,2]) - torch.tensor(0.5).to(self.device)
        self.beta_M = torch.tensor(10).to(self.device) + self.sig(params[:,3])*torch.tensor(30).to(self.device)
        self.M_cr = self.tanh(params[:,4])
        self.lnA = params[:,5]
        self.EaR = self.rel(params[:,7])
        self.n = self.sig(params[:,8])
        self.crit_shear = self.tanh(params[:,9])
        self.beta_shear = torch.tensor(10).to(self.device) + self.sig(params[:,10])*torch.tensor(30).to(self.device)
        
        #Temp
        a_t = self.lnA + self.EaR*T

        if (a_t > 2).any():
            print('caught invalid temp shift')
            filter_idx = (a_t > 2).nonzero(as_tuple=True)
            for i in filter_idx:
                print('lnA', self.lnA[i], 'EaR', self.EaR[i], 'T', T[i], 'T_shift', a_t[i])
        
        #NEW
        eta_0 = self.Mw_layer(M, self.alpha_1, self.alpha_2, self.beta_M, self.M_cr, self.k_1 + a_t)
        eta = self.Shear_layer(S, eta_0, self.n, self.beta_shear, self.crit_shear)

        eta = torch.unsqueeze(eta, -1)

        if torch.isnan(eta).any():
            print('invalid out')
            nan_idx = (torch.isnan(eta)==1).nonzero(as_tuple=True)
            for i in nan_idx:
                print('Mcr', self.M_cr[i],'a2-a1' ,(self.alpha_2 - self.alpha_1)[i], 'crit_shear', self.crit_shear[i],
                'S', S, 'M', M, 'T', T)

        if train:
            return eta, self.alpha_1, self.alpha_2
        elif get_constants:
            return {Visc_Constants.a1.value : self.alpha_1, 
                    Visc_Constants.a2.value :    self.alpha_2, 
                    Visc_Constants.Mcr.value :    self.M_cr, 
                    Visc_Constants.k1.value :    self.k_1, 
                    Visc_Constants.lnA.value :    self.lnA, 
                    Visc_Constants.EaR.value :    self.EaR, 
                    Visc_Constants.Scr.value :    self.crit_shear, 
                    Visc_Constants.n.value :    self.n,
                    Visc_Constants.beta_M.value :    self.beta_M,
                    Visc_Constants.beta_shear.value :    self.beta_shear}
        else:
            return eta

class Visc_PENN_PI_WLF(Visc_PENN_Base):
    def __init__(self, n_fp, config,device, apply_gradnorm = False, **kwargs):
        '''
        :param n_fp (int): input fingerprint dimension
        :param config (dict): configurations for hyperparameters for the MLP
        :param device: device, either cuda:X or cpu
        '''
        super(Visc_PENN_PI_WLF, self).__init__(n_fp, config,device, apply_gradnorm, **kwargs)

        self.mlp = MLP_PENN_PI(n_fp, config, latent_param_size=11)
    
    def forward(self, fp, M, S, T, PDI, train:bool = False, get_constants:bool = False):
        """
        Performs a forward pass through the PENN
        
        :param fp: The fingerprint vector of size n_fp
        :param M: The scaled log mol. weight of the polymer melt
        :param S: The scaled shear rate of the polymer melt 
        :param T: The scaled temperature of the polymer melt
        :param PDI: The scaled polydispersity index of the polymer melt
        :param train (bool): If its set to true
        A boolean parameter that indicates whether the forward pass is being performed during training or not. 
        If set to True, it returns all quantities that are needed for the loss calculation. In this case its the calcualted eta, alpha_1, and alpha_2
        :param get_constants (bool): Boolean flag that indicates whether or not to return the empirical constants calculated by the
        forward pass of the MLP. defaults to False (optional)
        """
        eta = None
        
        M, S = torch.squeeze(M), torch.squeeze(S)
        params = self.mlp(fp, PDI, T)
        self.alpha_1 = self.sig(params[:,0])*torch.tensor(3).to(self.device)
        self.alpha_2 = self.sig(params[:,1])*torch.tensor(6).to(self.device)
        self.k_1 = self.tanh(params[:,2]) - torch.tensor(0.5).to(self.device)
        self.beta_M = torch.tensor(10).to(self.device) + self.sig(params[:,3])*torch.tensor(30).to(self.device)
        self.M_cr = self.tanh(params[:,4])
        self.C_1 = params[:,5]#self.sig(params[:,5])*torch.tensor(2).to(self.device)
        self.C_2 = params[:,6]#self.sig(params[:,6])*torch.tensor(2).to(self.device)
        self.T_r = self.tanh(params[:,7]) - 0.5
        self.n = self.sig(params[:,8])
        self.crit_shear = self.tanh(params[:,9])
        self.beta_shear = self.sig(params[:,10])*torch.tensor(30).to(self.device)

        #NEW
        eta_0 = self.Mw_layer(M, self.alpha_1, self.alpha_2, self.beta_M, self.M_cr, self.k_1)
        eta = self.Shear_layer(S, eta_0, self.n, self.beta_shear, self.crit_shear)

        eta = torch.unsqueeze(eta, -1)

        if torch.isnan(eta).any():
            print('invalid out')
            nan_idx = (torch.isnan(eta)==1).nonzero(as_tuple=True)
            for i in nan_idx:
                print('Mcr', self.M_cr[i],'a2-a1' ,(self.alpha_2 - self.alpha_1)[i], 'crit_shear', self.crit_shear[i],
                'S', S, 'M', M, 'T', T)

        if train:
            return eta, self.alpha_1, self.alpha_2, self.C_1, self.C_2, self.T_r
        elif get_constants:
            return eta, self.alpha_1, self.alpha_2, self.M_cr, self.k_1 , self.C_1, self.C_2, self.T_r, self.crit_shear, self.n, eta_0
        else:
            return eta

    def train_step(self, XX, M, S, T, P, visc, criterion, optimizer) -> dict:
        
        out, a1,a2, c1,c2, tr = self.forward(XX, M, S, T, P, train = True)

        loss = criterion(out, visc)
        a1_loss = self.config["a_weight"]*criterion(a1, torch.ones_like(a1).to(self.device))
        a2_loss = self.config["a_weight"]*criterion(a2, torch.ones_like(a2)*torch.tensor(3.4).to(self.device)) 
        
        dvisc_dT = torch.autograd.grad(out, T, grad_outputs=torch.ones_like(out), create_graph=True)[0].squeeze()
        
        # print("dvisc/dT",dvisc_dT)
        # print("dvisc/dT shape",dvisc_dT.shape)
        # print("param calc d visc/dT", ((c1*c2)/((c2+(T-tr))**2)))
        # print("param calc d visc/dT shape", ((c1*c2)/((c2+(T-tr))**2)).shape)
        t_loss = 0.0000001*criterion(dvisc_dT, (-(c1*c2)/((c2+(T-tr))**2)))
        # print("loss", loss)
        # print("t_loss", t_loss)
        # print("loss shape", loss.shape)
        # print("a1_loss shape", a1_loss.shape)
        # print("a2_loss shape", a2_loss.shape)
        # print("T_loss shape", t_loss.shape)
        loss = loss + a2_loss + a1_loss + t_loss
        #print(loss.device)
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        return {LossTypes.tot_train.value : loss, LossTypes.a1.value : a1_loss, LossTypes.a2.value : a2_loss, LossTypes.phys.value: t_loss}

class Visc_PENN_PI_Arrhenius(Visc_PENN_Base):
    def __init__(self, n_fp, config,device, apply_gradnorm = False):
        '''
        :param n_fp (int): input fingerprint dimension
        :param config (dict): configurations for hyperparameters for the MLP
        :param device: device, either cuda:X or cpu
        '''
        super(Visc_PENN_PI_Arrhenius, self).__init__(n_fp, config,device, apply_gradnorm=apply_gradnorm)

        self.device = device
        self.mlp = MLP_PENN_PI(n_fp, config, latent_param_size=9).to(self.device)

    def forward(self, fp, M, S, T, PDI, train:bool = False, get_constants:bool = False):
        """
        Performs a forward pass through the PENN
        
        :param fp: The fingerprint vector of size n_fp
        :param M: The scaled log mol. weight of the polymer melt
        :param S: The scaled shear rate of the polymer melt 
        :param T: The scaled temperature of the polymer melt
        :param PDI: The scaled polydispersity index of the polymer melt
        :param train (bool): If its set to true
        A boolean parameter that indicates whether the forward pass is being performed during training or not. 
        If set to True, it returns all quantities that are needed for the loss calculation. In this case its the calcualted eta, alpha_1, and alpha_2
        :param get_constants (bool): Boolean flag that indicates whether or not to return the empirical constants calculated by the
        forward pass of the MLP. defaults to False (optional)
        """
        eta = None
        
        M, S = torch.squeeze(M), torch.squeeze(S)
        params = self.mlp(fp, PDI, T)
        self.alpha_1 = self.sig(params[:,0])*torch.tensor(3).to(self.device)
        self.alpha_2 = self.sig(params[:,1])*torch.tensor(6).to(self.device)
        self.k_1 = self.tanh(params[:,2])
        self.beta_M = torch.tensor(10).to(self.device) + self.sig(params[:,3])*torch.tensor(30).to(self.device)
        self.M_cr = self.tanh(params[:,4])
        self.EaR = params[:,5]
        self.n = self.sig(params[:,6])
        self.crit_shear = self.tanh(params[:,7])
        self.beta_shear = self.sig(params[:,8])*torch.tensor(30).to(self.device)

        #NEW
        eta_0 = self.Mw_layer(M, self.alpha_1, self.alpha_2, self.beta_M, self.M_cr, self.k_1)
        eta = self.Shear_layer(S, eta_0, self.n, self.beta_shear, self.crit_shear)

        eta = torch.unsqueeze(eta, -1)

        if torch.isnan(eta).any():
            print('invalid out')
            nan_idx = (torch.isnan(eta)==1).nonzero(as_tuple=True)
            for i in nan_idx:
                print('Mcr', self.M_cr[i],'a2-a1' ,(self.alpha_2 - self.alpha_1)[i], 'crit_shear', self.crit_shear[i],
                'S', S, 'M', M, 'T', T, 'S_sc', S_sc)

        if train:
            return eta, self.alpha_1, self.alpha_2, self.EaR
        elif get_constants:
            return eta, self.alpha_1, self.alpha_2, self.M_cr, self.k_1, self.lnA, self.EaR, self.crit_shear, self.n, eta_0
        else:
            return eta

    def train_step(self, XX, M, S, T, P, visc, criterion, optimizer) -> dict:
        
        out, a1,a2, E = self.forward(XX, M, S, T, P, train = True)

        loss = criterion(out, visc)
        a1_loss = self.config["a_weight"]*criterion(a1, torch.ones_like(a1).to(self.device))
        a2_loss = self.config["a_weight"]*criterion(a2, torch.ones_like(a2)*torch.tensor(3.4).to(self.device)) 
        
        dvisc_dT = torch.autograd.grad(out, T, grad_outputs=torch.ones_like(out), create_graph=True)[0].squeeze()
        
        # print("dvisc/dT",dvisc_dT)
        # print("dvisc/dT shape",dvisc_dT.shape)
        # print("param calc d visc/dT", ((c1*c2)/((c2+(T-tr))**2)))
        # print("param calc d visc/dT shape", ((c1*c2)/((c2+(T-tr))**2)).shape)
        t_loss = 0.0001*criterion(dvisc_dT, E)
        # print("loss", loss)
        # print("t_loss", t_loss)
        # print("loss shape", loss.shape)
        # print("a1_loss shape", a1_loss.shape)
        # print("a2_loss shape", a2_loss.shape)
        # print("T_loss shape", t_loss.shape)
        loss = loss + a2_loss + a1_loss + t_loss
        #print(loss.device)
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        return {LossTypes.tot_train.value : loss, LossTypes.a1.value : a1_loss, LossTypes.a2.value : a2_loss, LossTypes.phys.value: t_loss}

class MLP_PENN(nn.Module):
    def __init__(self, n_fp, config = {"l1" : 120, "l2" : 120, "d1" : 0.2, "d2" : 0.2}, latent_param_size = 11):
        super(MLP_PENN, self).__init__()
        # initialize hyperparameters
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
            self.out = nn.Linear(l2, latent_param_size)
            self.layers = nn.ModuleList([self.layer_1, self.d1,self.rel,self.layer_2, self.d2,self.rel, self.out])
        else:
            self.out = nn.Linear(l1, latent_param_size)
            self.layers = nn.ModuleList([self.layer_1,self.d1, self.rel, self.out])

    def forward(self, fp, PDI):
        out = None

        x = torch.cat((fp, PDI), 1)
        x = x.view(-1,self.n_fp + 1)
        for l in self.layers:
            x = l(x)
        out = x

        return out

class MLP_PENN_PI(nn.Module):
    def __init__(self, n_fp, config = {"l1" : 120, "l2" : 120, "d1" : 0.2, "d2" : 0.2}, latent_param_size = 11):
        super(MLP_PENN_PI, self).__init__()
        # initialize hyperparameters
        l1 = config["l1"]
        l2 = config["l2"]
        d1 = config["d1"]
        d2 = config["d2"]
        self.n_fp = n_fp
        self.layer_1 = nn.Linear(self.n_fp + 2, l1)
        self.rel = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.d1 = nn.Dropout(p = d1)
        if l2 > 0:    
            self.layer_2 = nn.Linear(l1, l2)
            self.d2 = nn.Dropout(p = d2)
            self.out = nn.Linear(l2, latent_param_size)
            self.layers = nn.ModuleList([self.layer_1, self.d1,self.rel,self.layer_2, self.d2,self.rel, self.out])
        else:
            self.out = nn.Linear(l1, latent_param_size)
            self.layers = nn.ModuleList([self.layer_1,self.d1, self.rel, self.out])

    def forward(self, fp, PDI, T):
        out = None

        x = torch.cat((fp, PDI, T), 1)
        x = x.view(-1,self.n_fp + 2)
        for l in self.layers:
            x = l(x)
        out = x

        return out

class MolWeight(nn.Module):
    def __init__(self, device):
        super(MolWeight, self).__init__()
        self.device = device
        self.HS = HeavySide(device)
    
    def forward(self, M, alpha_1, alpha_2, beta, Mcr, k_1):        
        low_mw = k_1 + alpha_1*M
        k_2 = k_1 + (alpha_1 - alpha_2)*Mcr
        high_mw = k_2 + (alpha_2)*M
        high_weight = self.HS(beta, M-Mcr)
        low_weight = self.HS(beta, Mcr-M)
        f_M = low_mw*low_weight + high_mw*high_weight
        
        if (f_M>4).any():
            print('invalid out in Mol weight func')
            nan_idx = (torch.isnan(f_M)>1).nonzero(as_tuple=True)
            print('a1', alpha_1, 'a2',alpha_2 ,'bM', beta, 'Mcr', Mcr, 'k_1', k_1, 'M', M, 'F(M)', f_M)
                
        return f_M

class ShearRate(nn.Module):
    def __init__(self, device):
        super(ShearRate, self).__init__()
        self.device = device
        self.HS = HeavySide(device)
    
    def forward(self, S, eta_0, n, beta, Scr):
        #f_S = eta_0 - torch.pow(beta, -1)*(n)*(torch.log(torch.tensor(1).to(self.device) + torch.exp(torch.tensor(-1).to(self.device)*S_sc)) + S_sc)
        print("shear S", S)
        print("shear equ n", n)
        print("shear equ crit_shear", Scr)
        print("shear eta_0", eta_0)
        low_shear = eta_0 
        high_shear = eta_0 - n*(S-Scr)
        high_weight = self.HS(beta, S-Scr)
        low_weight = self.HS(beta, Scr-S)
        print("shear high weight", high_weight)
        f_S = low_shear*low_weight + high_shear*high_weight

        if torch.isnan(f_S).any():
            print('invalid out in shear rate func')
            nan_idx = (torch.isnan(f_S)==1).nonzero(as_tuple=True)
            for i in nan_idx:
                print('eta0', eta_0[i], 'n',n[i], 'bshear', beta[i], 'crit_shear', Scr[i])
        # if (eta_0>2).any():
        #     print('eta0', eta_0, 'n',n, 'bshear', beta, 'crit_shear', Scr)
        
        print(f_S)
        return f_S

class HeavySide(nn.Module):
    def __init__(self, device):
        super(HeavySide, self).__init__()
        self.device = device
    
    def forward(self, beta, x):
        beta = torch.tensor(50).to(self.device)
        return 1 / (1+ torch.exp(-beta*x))

class MolWeight_softplus(nn.Module):
    def __init__(self, device):
        super(MolWeight_softplus, self).__init__()
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

class ShearRate_softplus(nn.Module):
    def __init__(self, device):
        super(ShearRate_softplus, self).__init__()
        self.device = device
    
    def forward(self, S, eta_0, n, beta, Scr):
        S_sc = beta*(S - Scr)
        f_S = eta_0 - torch.pow(beta, -1)*(n)*(torch.log(torch.tensor(1).to(self.device) + torch.exp(torch.tensor(-1).to(self.device)*S_sc)) + S_sc)
        
        if torch.isnan(f_S).any():
            print('invalid out')
            nan_idx = (torch.isnan(f_S)==1).nonzero(as_tuple=True)
            for i in nan_idx:
                print('eta0', eta_0[i], 'n',n[i], 'bshear', beta[i], 'Scr', Scr[i])
        if (eta_0>2).any():
            print('eta0', eta_0, 'n',n, 'bshear', beta, 'Scr', Scr)
        
        
        return f_S