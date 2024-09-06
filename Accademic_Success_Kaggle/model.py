import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np


class Model(nn.Module):

    def __init__(self):

        super().__init__()

        self.model=nn.Sequential(
            nn.Linear(37,100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(100,3),
            nn.Softmax(dim=1)
        )

    def forward(self,x):
        return self.model(x)








