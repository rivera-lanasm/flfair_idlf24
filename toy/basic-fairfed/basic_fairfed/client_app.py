"""basic-fairfed: A Flower / PyTorch app."""

import torch
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context

from basic_fairfed.task import (
    # model
    Net,
    # load data method
    load_data,
    # get parameters from trained model
    get_weights,
    # set parameters on model given model and parameter
    set_weights,
    # train method 
    train,
    # test method
    test,
)


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):

    def __init__(self, net, trainloader, valloader, local_epochs):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def fit(self, parameters, config):
        """
        trains the model using the local dataset
        """
        set_weights(self.net, parameters)
        train_loss = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            self.device,
        )
        return get_weights(self.net), len(self.trainloader.dataset), {"train_loss": train_loss}

    def evaluate(self, parameters, config):
        """
        evaluate the model received on a held-out validation set that the client might have:
        """
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    """
    context enables you to get access to hyperparemeters defined in your pyproject.toml
    
    """
    # Load model and data
    net = Net()
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions)
    local_epochs = context.run_config["local-epochs"]

    # Return Client instance
    return FlowerClient(net, trainloader, valloader, local_epochs).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
