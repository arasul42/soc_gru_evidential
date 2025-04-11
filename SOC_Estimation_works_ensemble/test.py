import torch
from torch import nn
from edl_pytorch import NormalInvGamma, evidential_regression

model = nn.Sequential(
    nn.Linear(1, 16),  # one input dim
    nn.ReLU(),
    NormalInvGamma(16, 1),  # one target variable
)

x = torch.randn(1, 1)  # (batch, dim)
y = torch.randn(1, 1)

pred_nig = model(x)  # (mu, v, alpha, beta)

loss = evidential_regression(
    pred_nig,      # predicted Normal Inverse Gamma parameters
    y,             # target labels
    lamb=0.001,    # regularization coefficient 
)