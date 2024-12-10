# Copyright 2020 Flower Labs GmbH. All Rights Reserved.
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

Paper: arxiv.org/abs/1602.05629
"""

import numpy as np
from functools import reduce
from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union
import json
import statistics

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

# from .aggregate import aggregate, aggregate_inplace, weighted_loss_avg
from flwr.server.strategy.aggregate import  weighted_loss_avg
from flwr.server.strategy.strategy import Strategy


WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""


def aggregate_fair(weights, results, beta, gamma) -> NDArrays:
    """Compute weighted average."""
    list_id = [id for params, num_examples, id, acc, eod, indf in results]
    # client EOD metrics
    list_eod = [eod for params, num_examples, id, acc, eod, indf in results]
    # client Indf metrics
    list_indf = [indf for params, num_examples, id, acc, eod, indf in results]
    # Aggregate (Global) EOD metric stats
    avg_eod = 0 if len(list_eod) == 0 else np.mean(list_eod)
    max_eod = 0 if len(list_eod) == 0 else max(list_eod)
    avg_indf = 0 if len(list_indf) == 0 else np.mean(list_indf)

    # client Accuracy metrics
    list_acc = [acc for params, num_examples, id, acc, eod, indf in results]
    # Aggregate (Global) Acc metric stats
    avg_acc = 0 if len(list_acc) == 0 else np.mean(list_acc)
    max_acc = 0 if len(list_acc) == 0 else max(list_acc)

    # capture client deltas
    client_deltas = {}
    for client, res in enumerate(results):
        params, num_examples, id, acc, eod, indf = res
        metric, avg_metric = eod, avg_eod 
        # Use accuracy when metric is NaN
        if np.isnan(metric):
            metric = acc
            avg_metric = avg_acc
        # Use np.nan_to_num to handle NaNs in delta calculation

        # 1) EOD delta
        delta_eod = abs(np.nan_to_num(metric - avg_metric, nan=0.0))
        # 2) ind fairness delta 
        delta_indf = abs(np.nan_to_num(indf - avg_indf, nan=0.0))
        # weighted avg delta
        client_deltas[id] = gamma*delta_indf + (1 - gamma)*delta_eod

    # average delta, equation 6
    ave_delta = np.mean(list(client_deltas.values()))

    # calculate new client parameter weights (unnormalized)
    for client, res in enumerate(results):
        # unpack 
        params, num_examples, id, acc, eod = res
        # rename metrics
        metric, avg_metric = eod, avg_eod
        # adjusted weight
        weight_update = -beta*(client_deltas[id] - ave_delta)
        if id in weights:
            weights[id] = weights[id] + weight_update
        else:
            weights[id] = weight_update

    # normalize updated weights, equation 6
    # Normalize updated weights, handling NaNs if present
    weight_norm_factor = np.sum(np.nan_to_num(list(weights.values()), nan=0.0))
    for key, val in weights.items():
        normalized_weight = np.nan_to_num(val / weight_norm_factor, nan=0.0)
        weights[key] = normalized_weight
        # new_weight.append(weights[key])

    # weighting the paramters for each client by new_weights
    weighted_weights = []
    for res in results:
        params, num_examples, id, acc, eod = res
        if id in weights:
            weighted_weights.append([layer * weights[id] for layer in params] )
        else:
            weighted_weights.append([layer * 0 for layer in params] )

    # weighted_weights = [
    #     [layer * new_weight[client] for layer in res[0]] for client, res in enumerate(results)
    # ]
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
    num_examples_total = sum([num_examples for params, num_examples, id, acc, eod in results])
    # Create a list of weights, each multiplied by the related number of examples
    weights = {}
    for params, num_examples, id, acc, eod in results:
        weights[id] = num_examples / num_examples_total
    return weights


# pylint: disable=line-too-long
class CustomFairFed(Strategy):
    """Federated Averaging strategy.

    Implementation based on https://arxiv.org/abs/1602.05629

    Parameters
    ----------
    fraction_fit : float, optional
        Fraction of clients used during training. In case `min_fit_clients`
        is larger than `fraction_fit * available_clients`, `min_fit_clients`
        will still be sampled. Defaults to 1.0.
    fraction_evaluate : float, optional
        Fraction of clients used during validation. In case `min_evaluate_clients`
        is larger than `fraction_evaluate * available_clients`,
        `min_evaluate_clients` will still be sampled. Defaults to 1.0.
    min_fit_clients : int, optional
        Minimum number of clients used during training. Defaults to 2.
    min_evaluate_clients : int, optional
        Minimum number of clients used during validation. Defaults to 2.
    min_available_clients : int, optional
        Minimum number of total clients in the system. Defaults to 2.
    evaluate_fn : Optional[Callable[[int, NDArrays, Dict[str, Scalar]],Optional[Tuple[float, Dict[str, Scalar]]]]]
        Optional function used for validation. Defaults to None.
    on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure training. Defaults to None.
    on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure validation. Defaults to None.
    accept_failures : bool, optional
        Whether or not accept rounds containing failures. Defaults to True.
    initial_parameters : Parameters, optional
        Initial global model parameters.
    fit_metrics_aggregation_fn : Optional[MetricsAggregationFn]
        Metrics aggregation function, optional.
    evaluate_metrics_aggregation_fn : Optional[MetricsAggregationFn]
        Metrics aggregation function, optional.
    inplace : bool (default: True)
        Enable (True) or disable (False) in-place aggregation of model updates.
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes, line-too-long
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 10,
        min_evaluate_clients: int = 10,
        min_available_clients: int = 10,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        inplace: bool = True,
        # betas
        beta: float,
        # gamma
        gamma: float
    ) -> None:
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
        self.inplace = inplace
        self.beta = beta
        self.gamma = gamma

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"FedAvg(accept_failures={self.accept_failures})"
        return rep

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return the sample size and the required number of available clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

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

        print("LOGG: RESULTS")
        for res in results:
            print(res[1].metrics, res[1].num_examples)

        # Convert results with NaN handling for each individual array
        weights_results = [
            (
                [np.nan_to_num(layer, nan=0.0) for layer in parameters_to_ndarrays(fit_res.parameters)],  # Handle NaNs in each layer
                fit_res.num_examples,
                fit_res.metrics['id'],
                np.nan_to_num(fit_res.metrics['acc'], nan=0.0),  # Replace NaNs in accuracy
                np.nan_to_num(fit_res.metrics['eod'], nan=0.0),   # Replace NaNs in EOD
                np.nan_to_num(fit_res.metrics['indf'], nan=0.0),   # Replace NaNs in IndF
                
            )
            for _, fit_res in results
        ]

        if not weights_results:
            return None, {}


        # Initialize current_weights in first server round
        if server_round == 1:
            self.current_weights = fedavg_weights(weights_results)

        # print("============= LOG WEIGHT RESULTS")
        # print([val[2] for val in weights_results])
        # print(self.current_weights)

        # weighted average of client parameters
        weights, parameters_aggregated = aggregate_fair(weights = self.current_weights, 
                                                        results = weights_results, 
                                                        beta = self.beta, 
                                                        gamma = self.gamma)
        parameters_aggregated = ndarrays_to_parameters(parameters_aggregated)

        # weights is returned by aggregate_fair so we can do this update
        self.current_weights = weights

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

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