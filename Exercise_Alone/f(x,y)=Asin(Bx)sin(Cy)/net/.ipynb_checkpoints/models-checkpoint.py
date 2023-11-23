import torch
import torch.nn as nn
import torch.nn.functional as F

from .prune import PruningModule, MaskedLinear

class NN(PruningModule):
    def __init__(self, mask=False):
        super(NN, self).__init__()
        linear = MaskedLinear if mask else nn.Linear
        self.fc1 = linear(2, 20)
        self.fc2 = linear(20, 20)
        self.fc3 = linear(20, 20)
        self.fc4 = linear(20, 20)
        self.fc5 = linear(20, 20)
        self.fc6 = linear(20, 20)
        self.fc7 = linear(20, 20)
        self.fc8 = linear(20, 1)
        
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = torch.tanh(self.fc5(x))
        x = torch.tanh(self.fc6(x))
        x = torch.tanh(self.fc7(x))
        x = self.fc8(x)
        return x