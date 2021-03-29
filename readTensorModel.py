import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt
import pickle

model = nn.Linear(1, 1).to(device='cuda:0')


model.load_state_dict(torch.load('LR_sample.torch'))

print('mdoel evalutation :',model.eval())


x = torch.tensor([0.56],dtype=torch.float32, device='cuda:0')

print(model(x).detach().item())