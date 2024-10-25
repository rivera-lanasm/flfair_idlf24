"""Custom Task Class."""

import json
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from flwr_datasets.partitioner import DirichletPartitioner

from flwr.common.typing import UserConfig

class Net(nn.Module):
    """Model (simple CNN adapted for Fashion-MNIST)"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

def compute_eod(preds, labels, sensitive_feature):
    preds_binary = (preds >= 0.5).float()
    y_true_mask = (labels == 1).view(-1)

    p_a0 = preds_binary[y_true_mask & (sensitive_feature == 0)].mean().item()
    p_a1 = preds_binary[y_true_mask & (sensitive_feature == 1)].mean().item()

    eod = p_a0 - p_a1
    return eod

def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set and calculate EOD."""
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    net.train()
    running_loss = 0.0
    
    # Store predictions and sensitive features for EOD calculation
    all_preds, all_labels, all_sensitives = [], [], []
    
    for _ in range(epochs):
        for inputs, labels, sensitive_features in trainloader:  # Include sensitive feature in loop
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            labels = labels.view(-1, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Store outputs, labels, and sensitive attributes for EOD
            all_preds.append(outputs.detach().cpu())
            all_labels.append(labels.detach().cpu())
            all_sensitives.append(sensitive_features.cpu())
    
    # Concatenate all batches for EOD calculation
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_sensitives = torch.cat(all_sensitives)
    
    # Calculate EOD
    eod = compute_eod(all_preds, all_labels, all_sensitives)
    avg_trainloss = running_loss / len(trainloader)
    print(f"Avg Train Loss: {avg_trainloss} - EOD: {eod}")
    return avg_trainloss, eod

def test(net, testloader, device):
    """Validate the model on the test set and calculate EOD."""
    net.to(device)
    criterion = nn.BCELoss()
    correct, loss, total = 0, 0, 0
    all_preds, all_labels, all_sensitives = [], [], []
    
    with torch.no_grad():
        for inputs, labels, sensitive_features in testloader:  # Include sensitive feature in loop
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            labels = labels.view(-1, 1)
            loss += criterion(outputs, labels).item() * inputs.size(0)
            predicted = (outputs >= 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Store outputs, labels, and sensitive attributes for EOD
            all_preds.append(outputs.detach().cpu())
            all_labels.append(labels.detach().cpu())
            all_sensitives.append(sensitive_features.cpu())

    # Calculate accuracy and average loss
    accuracy = correct / total
    avg_loss = loss / len(testloader)
    
    # Concatenate for EOD calculation
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_sensitives = torch.cat(all_sensitives)
    
    # Calculate EOD
    eod = compute_eod(all_preds, all_labels, all_sensitives)
    print(f"Test Accuracy: {accuracy} - Test Loss: {avg_loss} - EOD: {eod}")
    return avg_loss, accuracy, eod

def get_weights(net):
    """
    helper function: get the updated model parameters from the local model
    """
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    """
    helper function: update the local model with parameters received from the server
    
    """
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


fds = None  # Cache FederatedDataset

def create_run_dir(config: UserConfig) -> Path:
    """Create a directory where to save results from this run."""
    # Create output directory given current timestamp
    current_time = datetime.now()
    run_dir = current_time.strftime("%Y-%m-%d/%H-%M-%S")
    # Save path is based on the current directory
    save_path = Path.cwd() / f"outputs/{run_dir}"
    save_path.mkdir(parents=True, exist_ok=False)

    # Save run config as json
    with open(f"{save_path}/run_config.json", "w", encoding="utf-8") as fp:
        json.dump(config, fp)

    return save_path, run_dir
