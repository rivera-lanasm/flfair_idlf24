"""Custom Task Class."""

import json
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from flwr_datasets.partitioner import DirichletPartitioner
from datasets import Dataset
import pandas as pd
from flwr.common.typing import UserConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

def load_data():
    global train_loaders, val_loaders, test_loader

    if 'trainloaders' in globals():
        return train_loaders, val_loaders, test_loader
    df = pd.read_csv('.kaggle/data/propublicaCompassRecividism_data_fairml.csv/propublica_data_for_fairml.csv')
    df['caucasian'] = ((df['African_American'] + df['Asian'] + df['Hispanic'] + df['Native_American'] + df['Other']) == 0).astype(int)
    NUM_CLIENTS = 10
    trainset, testset = train_test_split(df, test_size=0.2)
    batch_size = 32

    ds = Dataset.from_pandas(trainset)
    partitioner = DirichletPartitioner(
        num_partitions=NUM_CLIENTS,
        partition_by="caucasian",
        alpha=0.5,
        min_partition_size=(len(trainset) // (4 * NUM_CLIENTS)),
        self_balancing=True,
        shuffle=True
    )

    partitioner.dataset = ds
    datasets = []
    for i in range(NUM_CLIENTS):
        curr_partition = partitioner.load_partition(i)
        datasets.append(curr_partition.to_pandas())

    train_loaders = []
    val_loaders = []

    feature_columns = ['Number_of_Priors', 'score_factor','Age_Above_FourtyFive', 'Age_Below_TwentyFive', 'Misdemeanor']

    for ds in datasets:
        train_x = ds[feature_columns].values
        train_y = ds['Two_yr_Recidivism'].values
        sensitive_feature = ds['caucasian'].values

        train_x, val_x, train_y, val_y, sensitive_train, sensitive_val = train_test_split(
            train_x, train_y, sensitive_feature, test_size=0.25, shuffle=True, stratify=train_y, random_state=42
        )
        
        train_x_tensor = torch.from_numpy(train_x).float()
        train_y_tensor = torch.from_numpy(train_y).float()
        sensitive_train_tensor = torch.from_numpy(sensitive_train).float()

        valid_x_tensor = torch.from_numpy(val_x).float()
        valid_y_tensor = torch.from_numpy(val_y).float()
        sensitive_val_tensor = torch.from_numpy(sensitive_val).float()

        # Create TensorDataset and DataLoader, including the sensitive attribute
        train_dataset = TensorDataset(train_x_tensor, train_y_tensor, sensitive_train_tensor)
        valid_dataset = TensorDataset(valid_x_tensor, valid_y_tensor, sensitive_val_tensor)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(valid_dataset, batch_size=batch_size)

        train_loaders.append(train_loader)
        val_loaders.append(val_loader)

    # For test data
    test_x = testset[feature_columns].values
    test_y = testset['Two_yr_Recidivism'].values
    sensitive_test = testset['caucasian'].values

    test_x_tensor = torch.from_numpy(test_x).float()
    test_y_tensor = torch.from_numpy(test_y).float()
    sensitive_test_tensor = torch.from_numpy(sensitive_test).float()

    test_dataset = TensorDataset(test_x_tensor, test_y_tensor, sensitive_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loaders, val_loaders, test_loader

def get_train_data(partition_id):
    train_loaders, _, _ = load_data()
    return train_loaders[partition_id]

def get_val_data(partition_id):
    _, val_loaders, _ = load_data()
    return val_loaders[partition_id]

def get_test_data():
    _, _, test_loader = load_data()
    return test_loader

class Net(nn.Module):
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
