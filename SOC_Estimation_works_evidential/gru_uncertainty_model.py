import torch
import torch.nn as nn
import torch.nn.functional as F
from edl_pytorch import NormalInvGamma

class GRUEvidentialModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        super(GRUEvidentialModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, 256)
        self.selu1 = nn.SELU()
        self.fc2 = nn.Linear(256, 128)
        self.selu2 = nn.SELU()

        # EDL output layer (outputs gamma, nu, alpha, beta)
        self.edl = NormalInvGamma(128, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]  # Use the last time step
        out = self.selu1(self.fc1(out))
        out = self.selu2(self.fc2(out))
        
        gamma, nu, alpha, beta = self.edl(out)  # Outputs are already passed through softplus etc.
        return gamma, nu, alpha, beta




