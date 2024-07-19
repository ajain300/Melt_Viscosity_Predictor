import torch
import torch.nn as nn

from Melt_Viscosity_Predictor.utils.gradnorm import GradNorm




device = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# in init


output_grad = (torch.ones((32,11), device = device),)
input_grad = (torch.ones((32,64), device = device),)

print(output_grad[0])
print(input_grad[0])

dloss = output_grad[0].clone().detach().requires_grad_(True)
dl_loss_input = input_grad[0].clone().detach().requires_grad_(True)

print(dloss)
print(dl_loss_input)




