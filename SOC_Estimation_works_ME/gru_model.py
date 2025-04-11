import torch
import torch.nn as nn

class GRUModel(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, 256)  # Equivalent to first Dense(256)
        self.selu1 = nn.SELU()
        self.fc2 = nn.Linear(256, 128)  # Equivalent to Dense(128)
        self.selu2 = nn.SELU()
        self.fc3 = nn.Linear(128, output_size)  # Final output layer

    def forward(self, x):

        out, _ = self.gru(x)  # GRU layer processing
        out = self.fc1(out[:, -1, :])  # Fully connected layer (256 neurons)
        out = self.selu1(out)
        out = self.fc2(out)  # Fully connected layer (128 neurons)
        out = self.selu2(out)
        return self.fc3(out)  # Final output layer

# Standalone test for module execution
if __name__ == "__main__":
    # Example usage
    model = GRUModel(input_size=2, hidden_size=100, num_layers=2, output_size=1)
    print(model)


