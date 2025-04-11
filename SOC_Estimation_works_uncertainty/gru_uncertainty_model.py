import torch
import torch.nn as nn
import torch.nn.functional as F

class GRUEvidentialModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=4):
        super(GRUEvidentialModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, 256)
        self.selu1 = nn.SELU()
        self.fc2 = nn.Linear(256, 128)
        self.selu2 = nn.SELU()
        self.fc3 = nn.Linear(128, output_size)  # Output: gamma, nu, alpha, beta

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]  # Use last time step
        out = self.selu1(self.fc1(out))
        out = self.selu2(self.fc2(out))
        evidential_output = self.fc3(out)

        # Map to evidential parameters
        gamma = evidential_output[:, 0:1]                    # Mean prediction
        nu = F.softplus(evidential_output[:, 1:2])           # Evidence for precision
        alpha = F.softplus(evidential_output[:, 2:3]) + 1    # Shape (>1)
        beta = F.softplus(evidential_output[:, 3:4])         # Scale

        return gamma, nu, alpha, beta



