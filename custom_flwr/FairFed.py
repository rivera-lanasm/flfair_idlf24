# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Federated Averaging (FedAvg) [McMahan et al., 2016] strategy.

Paper: https://arxiv.org/abs/1602.05629



CODE FROM: https://github.com/healthylaife/FairFedAvg/blob/main/FairFedAvg.py
"""

import json
from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from flwr.server.strategy.aggregate import  weighted_loss_avg
from flwr.server.strategy.strategy import Strategy

from functools import reduce
import numpy as np
import statistics


WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""

def aggregate_fair(weights, results, beta) -> NDArrays:
    """Compute weighted average."""
    # client EOD metrics
    list_eod = [eod for params, num_examples, id, acc, eod in results]
    # Aggregate (Global) EOD metric stats
    avg_eod = 0 if len(list_eod) == 0 else np.mean(list_eod)
    max_eod = 0 if len(list_eod) == 0 else max(list_eod)

    # client Accuracy metrics
    list_eod = [eod for params, num_examples, id, acc, eod in results]
    # Aggregate (Global) EOD metric stats
    avg_eod = 0 if len(list_eod) == 0 else np.mean(list_eod)
    max_eod = 0 if len(list_eod) == 0 else max(list_eod)

    # calculate new client parameter weights
    new_weight = []
    for client, res in enumerate(results):
        # unpack 
        params, num_examples, id, acc, eod = res
        # rename metrics
        metric, avg_metric, max_metric = eod, avg_eod, max_eod 
        # TODO --> is this a mistake? don't we use accuracy when metric not avail
        metric = metric if metric > 0 else avg_metric
        # is this correct imple,entation of delta?
        weights[id] += beta * (max_metric-metric)
        new_weight.append(weights[id])
    # for each client
    for id in weights:
        # this is equation 6
        weights[id] /= sum(new_weight)
    new_weight = [w / sum(new_weight) for w in new_weight]

    # weighting the paramters for each client by new_weights
    weighted_weights = [
        [layer * new_weight[client] for layer in res[0]] for client, res in enumerate(results)
    ]
    # 
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) for layer_updates in zip(*weighted_weights)
    ]
    return weights, weights_prime

def fedavg_weights(results: List[Tuple[NDArrays, int]]) -> NDArrays:
    """
    INITIALIZE weight as num_examples_i / total number examples
    """
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for params, num_examples, id, acc, eosd, wtpr, apsd in results])
    # Create a list of weights, each multiplied by the related number of examples
    weights = {}
    for params, num_examples, id, acc, eosd, wtpr, apsd in results:
        weights[id] = num_examples / num_examples_total

    return weights

def aggregate(results: List[Tuple[NDArrays, int]]) -> NDArrays:
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples, acc, sens_acc in results])

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples, acc, sens_acc in results
    ]

    # Compute average weights of each layer
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime



class FairFedAvg(Strategy):
    """Configurable FedAvg strategy implementation."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        *,
        # how many clients participate
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        # optional functions for custom evaluation and configuration of training and evaluation rounds.
        evaluate_fn: Optional[
                Callable[
                    [int, NDArrays, Dict[str, Scalar]],
                    Optional[Tuple[float, Dict[str, Scalar]]],
                ]
            ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        # decides whether the strategy should continue in case of client failures,
        accept_failures: bool = True,
        # sets the initial model parameters for federated learning.
        initial_parameters: Optional[Parameters] = None,
        # custom aggregation of metrics from clients.
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    ) -> None:
        """Federated Averaging strategy.

        Implementation based on https://arxiv.org/abs/1602.05629

        Parameters
        ----------
        fraction_fit : float, optional
            Fraction of clients used during training. Defaults to 1.0.
        fraction_evaluate : float, optional
            Fraction of clients used during validation. Defaults to 1.0.
        min_fit_clients : int, optional
            Minimum number of clients used during training. Defaults to 2.
        min_evaluate_clients : int, optional
            Minimum number of clients used during validation. Defaults to 2.
        min_available_clients : int, optional
            Minimum number of total clients in the system. Defaults to 2.
        evaluate_fn : Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]]
            ]
        ]
            Optional function used for validation. Defaults to None.
        on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure training. Defaults to None.
        on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure validation. Defaults to None.
        accept_failures : bool, optional
            Whether or not accept rounds containing failures. Defaults to True.
        initial_parameters : Parameters, optional
            Initial global model parameters.
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn]
            Metrics aggregation function, optional.
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn]
            Metrics aggregation function, optional.
        """
        super().__init__()

        if (
            min_fit_clients > min_available_clients
            or min_evaluate_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.current_weights = {}
        f = open("log/training.txt", "w")

    def __repr__(self) -> str:
        rep = f"FedAvg(accept_failures={self.accept_failures})"
        return rep
    
    ###
    ### ---- CLIENT SAMPLING METHODS ----------
    ###
    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Returns 
            1) the sample size  
            2) the required number of available clients (specified by min_available_clients)
        ."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    ###
    ### ----  ----------
    ###

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""

        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples, fit_res.metrics['id'], fit_res.metrics['acc'], fit_res.metrics['eosd'], fit_res.metrics['wtpr'], fit_res.metrics['apsd'])
            for _, fit_res in results
        ]

        # ----------- LOGGIN REASONS -------------# 
        total_acc = 0
        total_acc_val = 0
        avg_fair = 0
        num_val = 0
        num = 0
        for _, client in results:
            total_acc += client.metrics['acc'] * client.num_examples
            total_acc_val += client.metrics['acc_val'] * client.metrics['num_val']
            num += client.num_examples
            num_val += client.metrics['num_val']
        print('Training Accuracy: ' + str(total_acc / num) + ', Validation Accuracy: ' + str(total_acc_val / num_val))
        # ----------- LOGGIN REASONS -------------# 

        if server_round == 1:
            self.current_weights = fedavg_weights(weights_results)
        # TODO --> is this a mistake?
        # self.current_weights = fedavg_weights(weights_results)

        # TODO --> aggregated fair
        weights, parameters_aggregated = aggregate_fair(self.current_weights, weights_results, 1)
        parameters_aggregated = ndarrays_to_parameters(parameters_aggregated)
        
        # weights is returned by aggregate_fair so we can do this update
        self.current_weights = weights

        # round checkpoint 
        if parameters_aggregated is not None:
            print(f"Saving round {server_round} aggregated_ndarrays...")
            np.savez(f"checkpoints/round-{server_round}-weights.npz", parameters_aggregated)
        
        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")
        f = open("log/training.txt", "a")
        f.write('ROUND ' + str(server_round) + '\n')
        f.write('Training Accuracy: ' + str(total_acc / num) + ', Validation Accuracy: ' + str(total_acc_val / num_val) + '\n')
        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated