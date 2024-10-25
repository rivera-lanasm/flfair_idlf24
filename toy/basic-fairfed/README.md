# basic-fairfed: A Flower / PyTorch app

## Install dependencies and project

```bash
pip install -e .
```

## Run with the Simulation Engine

In the `basic-fairfed` directory, use `flwr run` to run a local simulation:

```bash
flwr run .
```

## Run with the Deployment Engine

> \[!NOTE\]
> An update to this example will show how to run this Flower application with the Deployment Engine and TLS certificates, or with Docker.

## Components 

Referring to [this tutorial](https://flower.ai/docs/framework/tutorial-quickstart-pytorch.html)

#### Data Loader Methods under task.py/load_data()

1) Flower DataSet
- see documnetation [here](https://flower.ai/docs/datasets/)
- Flower Datasets (flwr-datasets) is a library that enables the quick and easy creation of datasets for federated learning/analytics/evaluation. 
- It enables heterogeneity (non-iidness) simulation and division of datasets with the preexisting notion of IDs.

- pytorch specific [documentation](https://flower.ai/docs/datasets/how-to-use-with-pytorch.html)

2) Flower Partitioner
- https://flower.ai/docs/datasets/ref-api/flwr_datasets.partitioner.html

3) Client
- https://flower.ai/docs/framework/ref-api/flwr.client.Client.html#client
- **evaluation strategies**
    - https://flower.ai/docs/framework/explanation-federated-evaluation.html

4) Server
- https://flower.ai/docs/framework/ref-api/flwr.server.ServerAppComponents.html#serverappcomponents
- strategy is an **attribute** of this class
    - https://flower.ai/docs/framework/ref-api/flwr.server.strategy.Strategy.html#



## Advanced Pytorch Example

Documentation [here](https://flower.ai/docs/examples/advanced-pytorch.html#)
- this includes a custom strategy

