"""Custom ServerApp Class

Similar to ClientApp, we create a ServerApp using a utility function server_fn. 
In server_fn, we pass an instance of ServerConfig for defining the number of 
federated learning rounds (num_rounds) and we also pass the previously 
created strategy. 

The server_fn returns a ServerAppComponents object containing the settings 
that define the ServerApp behaviour. ServerApp is the entrypoint 
that Flower uses to call all your server-side code (for example, the strategy)

"""

import torch
# from custom_flwr.strategy import FairFed
from custom_flwr.idl24_FairFed import CustomFairFed as FairFed

from custom_flwr.task import (
    Net,
    get_weights,
    set_weights,
    test,
    get_test_data
)
from torch.utils.data import DataLoader

from datasets import load_dataset
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

BETA = 0.1
GAMMA = .5

def gen_evaluate_fn(testloader: DataLoader,
                    device: torch.device):
    """Generate the function for centralized evaluation."""

    def evaluate(server_round, parameters_ndarrays, config):
        """Evaluate global model on centralized test set."""
        net = Net()
        set_weights(net, parameters_ndarrays)
        net.to(device)
        loss, accuracy, eod, indf = test(net, testloader, device=device)
        return loss, {"centralized_accuracy": accuracy, "eod": eod, "indf":indf}

    return evaluate


def on_fit_config(server_round: int):
    """Construct `config` that clients receive when running `fit()`"""
    lr = 0.1
    # Enable a simple form of learning rate decay
    if server_round > 10:
        lr /= 2
    return {"lr": lr}


# Define metric aggregation function
def weighted_average(metrics):
    """
    Metric aggregation function
    
    """
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"federated_evaluate_accuracy": sum(accuracies) / sum(examples)}


def server_fn(context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    fraction_eval = context.run_config["fraction-evaluate"]
    server_device = context.run_config["server-device"]

    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)
    testloader = get_test_data()

    # Define strategy
    strategy = FairFed(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_eval,
        initial_parameters=parameters,
        on_fit_config_fn=on_fit_config,
        evaluate_fn=gen_evaluate_fn(testloader, device=server_device),
        evaluate_metrics_aggregation_fn=weighted_average,
        # weight on fairness
        beta=BETA,
        # weight on ind fairness
        gamma = GAMMA
    )

    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, 
                               config=config)

