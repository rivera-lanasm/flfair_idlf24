"""
pytorch-example: A Flower / PyTorch app
"""

import torch
from custom_flwr.task import Net, get_weights, set_weights, test, train, get_train_data, get_val_data

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, ParametersRecord, RecordSet, array_from_numpy


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    """A simple client that showcases how to use the state.

    It implements a basic version of `personalization` by which
    the classification layer of the CNN is stored locally and used
    and updated during `fit()` and used during `evaluate()`.
    """

    def __init__(
        self, net, client_state: RecordSet, trainloader, valloader, local_epochs
    ):
        self.net: Net = net
        self.client_state = client_state
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.local_layer_name = "classification-head"

    def fit(self, parameters, config):
        """Train model locally.

        The client stores in its context the parameters of the last layer in the model
        (i.e. the classification head). The classifier is saved at the end of the
        training and used the next time this client participates.

        Receive model parameters from the server, train the model on the local
          data, and return the updated model parameters to the server
        """

        # Apply weights from global models (the whole model is replaced)
        set_weights(self.net, parameters)

        # Override weights in classification layer with those this client
        # had at the end of the last fit() round it participated in
        self._load_layer_weights_from_state()

        train_loss, eod, acc = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            lr=float(config["lr"]),
            device=self.device,
        )
        # Save classification head to context's state to use in a future fit() call
        self._save_layer_weights_to_state()

        # Return locally-trained model and metrics
        return (
            # model paramters
            get_weights(self.net),
            # size of client data 
            len(self.trainloader.dataset),
            # metrics
            {"train_loss": train_loss, "eod": eod, "acc": acc},
        )

    def _save_layer_weights_to_state(self):
        """Save last layer weights to state."""
        state_dict_arrays = {}
        for k, v in self.net.fc2.state_dict().items():
            state_dict_arrays[k] = array_from_numpy(v.cpu().numpy())

        # Add to recordset (replace if already exists)
        self.client_state.parameters_records[self.local_layer_name] = ParametersRecord(
            state_dict_arrays
        )

    def _load_layer_weights_from_state(self):
        """Load last layer weights to state."""
        if self.local_layer_name not in self.client_state.parameters_records:
            return

        state_dict = {}
        param_records = self.client_state.parameters_records
        for k, v in param_records[self.local_layer_name].items():
            state_dict[k] = torch.from_numpy(v.numpy())

        # apply previously saved classification head by this client
        self.net.fc2.load_state_dict(state_dict, strict=True)

    def evaluate(self, parameters, config):
        """Evaluate the global model on the local validation set.

        Note the classification head is replaced with the weights this client had the
        last time it trained the model.

        Receive model parameters from the server, evaluate the model 
        on the local data, and return the evaluation result to the server
        """
        set_weights(self.net, parameters)
        # Override weights in classification layer with those this client
        # had at the end of the last fit() round it participated in
        self._load_layer_weights_from_state()
        loss, accuracy, eod = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy, "eod": eod}

def client_fn(context: Context):
    """
    
    special simulation capabilities that create FlowerClient instances only when
      they are actually necessary for training or evaluation. 
      
      To enable the Flower framework to create clients when necessary, 
      we need to implement a function that creates a FlowerClient instance on demand. 
      We typically call this function client_fn. 
      
      Flower calls client_fn whenever it needs an instance of one 
      particular client to call fit or evaluate (those instances are
        usually discarded after use, so they should not keep any local state)
    
    """
    # Load model and data
    net = Net()
    # local epochs
    local_epochs = 1
    trainloader = get_train_data(context.node_config['partition-id'])
    valloader = get_val_data(context.node_config['partition-id'])
    # Load data

    # Return Client instance
    # We pass the state to persist information across
    # participation rounds. Note that each client always
    # receives the same Context instance (it's a 1:1 mapping)
    client_state = context.state
    return FlowerClient(
        net, client_state, trainloader, valloader, local_epochs
    ).to_client()

# Flower ClientApp
app = ClientApp(
    client_fn,
)
