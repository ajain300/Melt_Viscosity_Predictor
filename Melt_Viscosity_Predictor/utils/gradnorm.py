#!/usr/bin/env python
# coding: utf-8
"""
Training with GradNorm Algorithm
"""

from wsgiref.util import request_uri
import numpy as np
import torch
import matplotlib.pyplot as plt


class GN(torch.nn.Module):
    def __init__(self, alpha, device, layer):
        super(GN, self).__init__()
        self.weights = torch.ones(layer.out_features, requires_grad=True,  device = device)
        self.T = self.weights.sum()
        self.weight_optimizer = torch.optim.Adam([self.weights], lr=0.001)
        self.lr = 10.0
        self.iter = 0
        self.l0 = None
        self.alpha = alpha

    def forward(self, output, input):
        # Element-wise multiplication expanded to batch
        # Expand weights to [batch_size, output_dim] by unsqueezing and using expand
        print("output_gradients", output[0, :])
        if self.iter == 0:
            self.l0 = output.abs().mean(axis = 0)
        self.output = output
        self.input = input
        expanded_weights = self.weights.unsqueeze(0).expand(output.size(0), -1)
        self.batch_size = output.size(0)
        self.exp_weights = expanded_weights
        # Multiply output with weights: [batch_size, output_dim]
        task_weighted_output_grad = output * expanded_weights
        self.task_weighted_output_grad = task_weighted_output_grad
        # Prepare for batch matrix multiplication:
        # task_weighted_output_grad: [batch_size, output_dim]
        # input: [batch_size, input_dim]
        # We need to add a dimension to task_weighted_output_grad to perform batch matrix multiplication
        # task_weighted_output_grad.unsqueeze(-1): [batch_size, output_dim, 1]
        # input.unsqueeze(1): [batch_size, 1, input_dim]
        dl_dW = torch.bmm(task_weighted_output_grad.unsqueeze(2), input.unsqueeze(1))
        # dl_dW: [batch_size, output_dim, input_dim]
        # Now, reduce across batch dimension to get the summed gradient for each weight
        # Assuming we need the norm of this matrix across the batch dimension:
        dl_dW = torch.mean(dl_dW, dim=(0))
        self.dl_dW = dl_dW
        gw = torch.norm(dl_dW, dim=(1))
        print("GW manual", gw)
        self.gw = gw

        self.iter += 1
        return gw

    def calculate_gradients(self, grad_loss_vector):
        print("GW grad manual", grad_loss_vector)
        exp_loss = grad_loss_vector.unsqueeze(-1).expand(-1, self.input.shape[1])
        norm_grad = (exp_loss * gradient_l2_norm(self.dl_dW)).expand(self.batch_size,-1, self.input.shape[1])
        dl_dW_grad = torch.bmm(norm_grad, self.input.unsqueeze(2))
        weight_grad = self.output*dl_dW_grad.squeeze(2)
        self.weight_grad = weight_grad.mean(axis = (0))
        print("manual weight grad", self.weight_grad)
        

    def step_weights(self):
        self.weights = self.weights - self.weight_grad*self.lr
        self.weights = (self.weights / self.weights.sum() * self.T)
        print("updated weights", self.weights)

    def loss(self, output, gw):
        gw = self.gw
        # compute loss ratio per task
        loss_ratio = output.detach().abs().mean(axis = 0) / self.l0
        # compute the relative inverse training rate per task
        rt = loss_ratio / loss_ratio.mean()
        # compute the average gradient norm
        gw_avg = gw.mean()
        # compute the GradNorm loss
        constant = (gw_avg * rt ** self.alpha)
        print("constant", constant)
        return gw - constant 


def gradient_l2_norm(x):
    return x/torch.norm(x, dim = (1)).reshape(-1, 1)


class GradNorm:
    """
    Class for gradient normalization on the PENN
    """

    def __init__(self, layer, alpha,lr2 , device,log=False):
        self.alpha = alpha
        self.log = log
        self.layer = layer
        self.iter = 0
        # self.weights = torch.nn.Parameter(torch.ones(layer.out_features, requires_grad=True,  device = device))
        # self.T = torch.tensor(layer.out_features, device = device)
        # self.weight_optimizer = torch.optim.Adam([self.weights], lr=lr2)
        self.l0 = None
        self.log_weights = []
        self.log_loss = []
        self.lr2 = lr2
        self.device = device
        self.gradnorm_model = GN(alpha = alpha, device=device, layer = layer)
        self.melt_visc_params = ["alpha_1", "alpha_2", "k_1", "beta_M", "M_cr", "C_1", "C_2", "T_r", "n", "crit_shear", "beta_shear"]

    
    def gradNorm_layer(self, module, input_grad, output_grad):
        dloss = output_grad[0].clone().detach().requires_grad_(True)
        dl_loss_input = input_grad[0].clone().detach().requires_grad_(True)

        out = self.gradnorm_model(dloss, dl_loss_input)
        gradnorm_loss_manual = self.gradnorm_model.loss(dloss, out)
        self.gradnorm_model.calculate_gradients(gradnorm_loss_manual)
        self.gradnorm_model.step_weights()

        # log weights and loss
        self.log_weights.append(self.gradnorm_model.weights.detach().cpu().numpy().copy())

        # calculate the input grad 
        weighted_input_grad = self.gradnorm_model.task_weighted_output_grad @ [param for param in self.layer.parameters()][0]

        return (weighted_input_grad,)
    
    def gradNorm_layer_old(self, module, input_grad, output_grad):
        """
        Takes in a layer's losses, and transforms the output according to 
        
        Args:
            net (nn.Module): a multitask network with task loss
            layer (nn.Module): a layer of the full network where appling GradNorm on the weights
            alpha (float): hyperparameter of restoring force
            optimizer (DataLoader): optimizer for the MLP of the PENN
            lr1（float): learning rate of multitask loss
            lr2（float): learning rate of weights
            log (bool): flag of result log
        """
        # start loss transformation
        # initialization

        #torch.autograd.set_detect_anomaly(True)
        dloss = output_grad[0].clone().detach().requires_grad_(True)
        dl_loss_input = input_grad[0].clone().detach().requires_grad_(True)

        out = self.gradnorm_model(dloss, dl_loss_input)
        gradnorm_loss_manual = self.gradnorm_model.loss(dloss, out)
        self.gradnorm_model.calculate_gradients(gradnorm_loss_manual)
        self.gradnorm_model.step_weights()

        print("manually calcualted gradnorm weights, ", self.gradnorm_model.weights)

        dloss = output_grad[0].clone().detach().requires_grad_(True)
        dl_loss_input = input_grad[0].clone().detach().requires_grad_(True)

        if self.iter == 0:
            # init weights
            # weights = torch.ones((dloss.shape[1]), device = self.device, requires_grad=True)
            # weights = torch.nn.Parameter(weights)
            # T = weights.clone().sum().detach() # sum of weights
            # set optimizer for weights
            
            # set L(0)
            self.l0 = dloss.detach().abs().mean(dim = 0)
        # else:
        #     weights = self.weights.clone()
        #     weights.requires_grad_(True)
        #     weights = torch.nn.Parameter(weights)
        #     self.weight_optimizer = torch.optim.Adam([weights], lr=self.lr2)
        #     T = weights.clone().sum().detach() # sum of weights

        # STEP 1 - weighted loss updates on mlp
        # compute the weighted loss
        weighted_output_grad = torch.mul(self.weights, dloss)
        weighted_input_grad = weighted_output_grad @ [param for param in self.layer.parameters()][0]

        # if self.iter == 0:
        #     assert torch.norm(weighted_input_grad - dl_loss_input) < 1e-12, f"input grad incorrectly calculated, error norm at {torch.norm(weighted_input_grad - dl_loss_input)}"
        # backward pass for weighted task loss
        # weighted_loss.backward(retain_graph=True)
        
        # STEP 2 - update the gradnorm loss
        gw = []
        for i in range(dloss.shape[1]):
            task_weighted_output_grad = dloss[:, i] * self.weights[i]
            dl_dW_i = task_weighted_output_grad.unsqueeze(1) * dl_loss_input
            dl_dW_i = torch.mean(dl_dW_i, dim = 0)
            dl_dW_i.retain_grad()
            # retain_graph=True, create_graph=True, allow_unused=True)
            gw.append(torch.norm(dl_dW_i))

        gw = torch.stack(gw)
        gw.retain_grad()
        # compute loss ratio per task
        loss_ratio = dloss.detach().abs().mean(axis = 0) / self.l0
        # compute the relative inverse training rate per task
        rt = loss_ratio / loss_ratio.mean()
        # compute the average gradient norm
        gw_avg = gw.mean()
        # compute the GradNorm loss
        constant = (gw_avg * rt ** self.alpha)

        gradnorm_loss = torch.abs(gw - constant).sum()
        # clear gradients of weights
        self.weight_optimizer.zero_grad()
        # backward pass for GradNorm

        print("GW ", gw)
        #print("gradnorm weights grad", torch.autograd.grad(gradnorm_loss, self.weights, allow_unused=True))
        gradnorm_loss.backward(retain_graph=True)
        print("gradnorm weights grad", self.weights.grad)
        print(" GW grad", gw.grad)
        print("dw/dw grad ", dl_dW_i.grad )
        # log weights and loss
        self.log_weights.append(self.weights.detach().cpu().numpy().copy())
        self.log_loss.append(loss_ratio.detach().mean(dim = 0).cpu().numpy().copy())
        
        # update loss weights
        self.weight_optimizer.step()
        print("updated weights ", self.weights)
        # renormalize weights
        self.weights = (self.weights / self.weights.sum() * self.T)
        # update iters
        self.iter += 1
        out_hook = (weighted_input_grad,)
        torch.autograd.set_detect_anomaly(False)
        if self.iter == 3:
            exit()

        return out_hook

    # def gradNorm_layer(self, loss, mlp_optimizer, lr2):
    #     """
    #     Takes in a layer's losses, and transforms the output according to 
        
    #     Args:
    #         net (nn.Module): a multitask network with task loss
    #         layer (nn.Module): a layer of the full network where appling GradNorm on the weights
    #         alpha (float): hyperparameter of restoring force
    #         optimizer (DataLoader): optimizer for the MLP of the PENN
    #         lr1（float): learning rate of multitask loss
    #         lr2（float): learning rate of weights
    #         log (bool): flag of result log
    #     """
    #     # start loss transformation
    #     # initialization
    #     if self.iter == 0:
    #         # init weights
    #         weights = torch.ones_like(loss)
    #         weights = torch.nn.Parameter(weights)
    #         T = weights.sum().detach() # sum of weights
    #         # set optimizer for weights
    #         self.weight_optimizer = torch.optim.Adam([weights], lr=lr2)
    #         # set L(0)
    #         self.l0 = loss.detach()
       
    #     # STEP 1 - weighted loss updates on mlp
    #     # compute the weighted loss
    #     weighted_loss = weights @ loss
    #     # clear gradients of network
    #     mlp_optimizer.zero_grad()
    #     # backward pass for weighted task loss
    #     # weighted_loss.backward(retain_graph=True)
        
    #     # STEP 2 - update the gradnorm loss
    #     # compute the L2 norm of the gradients for each task
    #     gw = []
    #     for i in range(len(loss)):
    #         dl = torch.autograd.grad(weights[i]*loss[i], self.layer.parameters(), retain_graph=True, create_graph=True)[0]
    #         gw.append(torch.norm(dl))
    #     gw = torch.stack(gw)
    #     # compute loss ratio per task
    #     loss_ratio = loss.detach() / self.l0
    #     # compute the relative inverse training rate per task
    #     rt = loss_ratio / loss_ratio.mean()
    #     # compute the average gradient norm
    #     gw_avg = gw.mean().detach()
    #     # compute the GradNorm loss
    #     constant = (gw_avg * rt ** self.alpha).detach()
    #     gradnorm_loss = torch.abs(gw - constant).sum()
    #     # clear gradients of weights
    #     self.weight_optimizer.zero_grad()
    #     # backward pass for GradNorm
    #     gradnorm_loss.backward()
        
    #     # log weights and loss
    #     self.log_weights.append(weights.detach().cpu().numpy().copy())
    #     self.log_loss.append(loss_ratio.detach().cpu().numpy().copy())
        
    #     # update loss weights
    #     self.weight_optimizer.step()
    #     # renormalize weights
    #     weights = (weights / weights.sum() * T).detach()
    #     weights = torch.nn.Parameter(weights)
    #     self.weight_optimizer = torch.optim.Adam([weights], lr=lr2)
    #     # update iters
    #     self.iter += 1

    #     return weighted_loss

    def get_log(self):    
        """
        Returns the weights over every train step
        """
        return np.stack(self.log_weights)
    
    def get_current_weights(self):
        """
        Return the last value of weights
        """
        weights = self.gradnorm_model.weights.detach().cpu().numpy().copy()
        return {p : w for p, w in zip(self.melt_visc_params, weights)}

    def plot_gradnorm_stats(self, path):
        weights = self.get_log()
        plt.figure()
        for i in range(11):
            print(i)
            print(weights[:, i])
            plt.plot(list(range(weights.shape[0])), np.log10(weights[:, i]).tolist(), label = f"Param {i}")
        #plt.plot(list(range(weights.shape[0])), loss.tolist())
        plt.legend()
        plt.savefig(path)

# Helper funciton to get the learning rate of an optimizer
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

if __name__ == '__main__':
    device = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    layer = torch.nn.Linear(64, 11, device = device)
    

    gradnorm = GradNorm(layer = layer, alpha = 1.5,lr2 = 0.001 , device = device)

    for step in range(5):
        input_grad = (torch.rand((32,64), device = device),)
        output_grad = (torch.rand((32,11), device = device),)
        output_grad[0][:, :3] = output_grad[0][:, :3]*0.000001
        print(output_grad[0][0, :])
        gradnorm.gradNorm_layer_old(layer, input_grad, output_grad)
