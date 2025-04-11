import torch
import torch.nn as nn
from edl_pytorch import NormalInvGamma

class GRUEvidentialModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(GRUEvidentialModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, 256)
        self.selu1 = nn.SELU()
        self.fc2 = nn.Linear(256, 128)
        self.selu2 = nn.SELU()
        self.fc3=nn.Linear(128,2) 

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]  # Use last time step
        out = self.selu1(self.fc1(out))
        out = self.selu2(self.fc2(out))
        out = self.fc3(out)
        mu = out[:, 0:1]
        log_var = out[:, 1:2]
        return mu, log_var  # returns tensor of shape [B, 4]
