import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os
import json

class GRUTrainer:
    def __init__(self, model, features, target, val_features, val_target, seq_length=40, batch_size=128, learning_rate=0.001, num_epochs=200, save_dir="./saved_models", validation_split=0.2):
        self.model = model
        self.device = next(model.parameters()).device  # Get device from model
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.save_dir = save_dir
        self.best_val_loss = float("inf")  # Track best validation loss

        # Ensure save directory exists
        os.makedirs(self.save_dir, exist_ok=True)

        # Create an incremental experiment folder (exp1, exp2, etc.)
        self.exp_dir = self.get_next_experiment_folder(save_dir)

        # Create sequences from full dataset
        X_seq, y_seq = self.create_sequences(features, target, seq_length)

        # Convert to PyTorch tensors
        X_train_torch = torch.tensor(X_seq, dtype=torch.float32)
        y_train_torch = torch.tensor(y_seq, dtype=torch.float32).view(-1, 1)

        # Split data into training and validation sets
        self.train_dataset = TensorDataset(X_train_torch, y_train_torch)
        # train_size = int((1 - validation_split) * len(dataset))
        # val_size = len(dataset) - train_size
        # self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])

        # Create sequences from validation dataset
        X_val_seq, y_val_seq = self.create_sequences(val_features, val_target, seq_length)
        X_val_torch = torch.tensor(X_val_seq, dtype=torch.float32)
        y_val_torch = torch.tensor(y_val_seq, dtype=torch.float32).view(-1, 1)

        self.val_dataset = TensorDataset(X_val_torch, y_val_torch)  



        # Prepare dataloaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

        # Define loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Training history
        self.train_losses = []
        self.val_losses = []


    def get_next_experiment_folder(self, save_dir):
        """Find the next available `expX` folder inside `save_dir`."""
        existing_exps = [d for d in os.listdir(save_dir) if d.startswith("exp") and d[3:].isdigit()]
        exp_nums = [int(d[3:]) for d in existing_exps]
        next_exp_num = max(exp_nums, default=0) + 1
        exp_path = os.path.join(save_dir, f"exp{next_exp_num}")
        os.makedirs(exp_path, exist_ok=True)
        return exp_path

    def create_sequences(self, features, target, seq_length):
        sequences, targets = [], []
        for i in range(len(features) - seq_length):
            sequences.append(features[i:i + seq_length])
            targets.append(target[i + seq_length])
        return np.array(sequences), np.array(targets)

    def train(self):
        """Train the GRU model with validation and save the best model."""
        print(f"Using device: {self.device}")
        print(f"Saving experiment to: {self.exp_dir}")

        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0

            for batch_X, batch_y in self.train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            # Compute average training loss
            train_loss /= len(self.train_loader)
            self.train_losses.append(train_loss)

            # Validation phase
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in self.val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    val_loss += loss.item()

            val_loss /= len(self.val_loader)
            self.val_losses.append(val_loss)

            # Save the best model if validation loss improves
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_model(filename="best_gru_model.pth")

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{self.num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

        # Save the last model
        self.save_model(filename="last_gru_model.pth")

    def save_model(self, filename):
        """Save the model state_dict."""
        model_path = os.path.join(self.exp_dir, filename)
        torch.save(self.model.state_dict(), model_path)

    def save_scalers(self, scaler_features, scaler_target):
        """Save the trained scalers for normalization."""
        joblib.dump(scaler_features, os.path.join(self.exp_dir, "scaler_features.pkl"))
        joblib.dump(scaler_target, os.path.join(self.exp_dir, "scaler_target.pkl"))
        print("Scalers saved successfully!")

    def save_training_config(self):
        """Save training configuration as JSON."""
        config = {
            "sequence_length": self.seq_length,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "learning_rate": self.optimizer.param_groups[0]["lr"]
        }
        config_path = os.path.join(self.exp_dir, "training_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
        print(f"Training configuration saved at {config_path}")

    def plot_loss(self):
        """Plot training and validation loss over epochs in separate and combined plots."""
        # Plot training loss
        plt.figure(figsize=(4.75, 3.02))
        plt.plot(self.train_losses, label="Training Loss", color="blue")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Epochs")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.exp_dir, "training_loss_plot.png"),dpi=300)
        # plt.show()

        # Plot validation loss
        plt.figure(figsize=(4.75, 3.02))
        plt.plot(self.val_losses, label="Validation Loss", color="orange")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Validation Loss Over Epochs")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.exp_dir, "validation_loss_plot.png"),dpi=300)
        # plt.show()

        # Plot combined training and validation loss
        plt.figure(figsize=(4.75, 3.72))
        plt.plot(self.train_losses, label="Training Loss", color="blue")
        plt.plot(self.val_losses, label="Validation Loss", color="orange")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss Over Epochs")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.exp_dir, "combined_loss_plot.png"),dpi=300)
        # plt.show()




