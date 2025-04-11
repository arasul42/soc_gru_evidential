import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os
import json


# def evidential_loss(y, gamma, nu, alpha, beta, lambda_reg=10.0, eps=1e-6):
#     nu = torch.clamp(nu, min=eps)
#     alpha = torch.clamp(alpha, min=1.0 + eps)
#     beta = torch.clamp(beta, min=eps)

#     squared_error = (y - gamma) ** 2
#     twoBlambda = 2 * beta * (1 + nu)
#     log_pi = torch.log(torch.tensor(np.pi, device=nu.device))

#     nll = (
#         0.5 * (log_pi - torch.log(nu))
#         - alpha * torch.log(twoBlambda)
#         + (alpha + 0.5) * torch.log(nu * squared_error + twoBlambda)
#         + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
#     )

#     error = torch.abs(y - gamma)
#     evidence = 2 * nu + alpha
#     reg = error * evidence

#     return torch.mean(nll + lambda_reg * reg)


def nig_nll(gamma, v, alpha, beta, y):
    two_beta_lambda = 2 * beta * (1 + v)
    t1 = 0.5 * (torch.pi / v).log()
    t2 = alpha * two_beta_lambda.log()
    t3 = (alpha + 0.5) * (v * (y - gamma) ** 2 + two_beta_lambda).log()
    t4 = alpha.lgamma()
    t5 = (alpha + 0.5).lgamma()
    nll = t1 - t2 + t3 + t4 - t5
    return nll.mean()

def nig_reg(gamma, v, alpha, _beta, y):
    reg = (y - gamma).abs() * (2 * v + alpha)
    return reg.mean()


def evidential_loss(y, gamma, nu, alpha, beta, lambda_reg=10.0):
    return nig_nll(gamma, nu, alpha, beta, y) + lambda_reg * nig_reg(gamma, nu, alpha, beta, y)



class GRUEvidentialTrainer:
    def __init__(self, model, features, target, seq_length=40, batch_size=128, learning_rate=0.001, num_epochs=200, save_dir="./saved_models", validation_split=0.2, lambda_reg=10.0):
        self.model = model
        self.device = next(model.parameters()).device
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.save_dir = save_dir
        self.best_val_loss = float("inf")

        os.makedirs(self.save_dir, exist_ok=True)
        self.exp_dir = self.get_next_experiment_folder(save_dir)

        X_seq, y_seq = self.create_sequences(features, target, seq_length)
        X_train_torch = torch.tensor(X_seq, dtype=torch.float32)
        y_train_torch = torch.tensor(y_seq, dtype=torch.float32).view(-1, 1)

        dataset = TensorDataset(X_train_torch, y_train_torch)
        train_size = int((1 - validation_split) * len(dataset))
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

        self.criterion = evidential_loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.lambda_reg = lambda_reg  # Define lambda_reg as an instance variable

        self.train_losses = []
        self.val_losses = []

    def get_next_experiment_folder(self, save_dir):
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
        print(f"Using device: {self.device}")
        print(f"Saving experiment to: {self.exp_dir}")

        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0

            for batch_X, batch_y in self.train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                self.optimizer.zero_grad()
                gamma, nu, alpha, beta = self.model(batch_X)
                loss = self.criterion(batch_y, gamma, nu, alpha, beta, lambda_reg=self.lambda_reg)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            train_loss /= len(self.train_loader)
            self.train_losses.append(train_loss)

            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in self.val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    gamma, nu, alpha, beta = self.model(batch_X)
                    loss = self.criterion(batch_y, gamma, nu, alpha, beta)
                    val_loss += loss.item()

            val_loss /= len(self.val_loader)
            self.val_losses.append(val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_model(filename="best_gru_model.pth")

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{self.num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

        self.save_model(filename="last_gru_model.pth")

    def save_model(self, filename):
        model_path = os.path.join(self.exp_dir, filename)
        torch.save(self.model.state_dict(), model_path)
        # print(f"Model saved at {model_path}")

    def save_scalers(self, scaler_features, scaler_target):
        joblib.dump(scaler_features, os.path.join(self.exp_dir, "scaler_features.pkl"))
        joblib.dump(scaler_target, os.path.join(self.exp_dir, "scaler_target.pkl"))
        print(f"Training configuration saved at {self.exp_dir}")

    def save_training_config(self):
        config = {
            "sequence_length": self.seq_length,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "lambda_reg": self.lambda_reg
        }
        config_path = os.path.join(self.exp_dir, "training_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
        print(f"Training configuration saved at {config_path}")

    def plot_loss(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label="Training Loss", color="blue")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Epochs")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.exp_dir, "training_loss_plot.png"))
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(self.val_losses, label="Validation Loss", color="orange")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Validation Loss Over Epochs")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.exp_dir, "validation_loss_plot.png"))
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label="Training Loss", color="blue")
        plt.plot(self.val_losses, label="Validation Loss", color="orange")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss Over Epochs")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.exp_dir, "combined_loss_plot.png"))
        plt.show()
