"""Custom FedAvg Class"""

import json
from logging import INFO

import torch
import wandb
from custom_flwr.task import Net, create_run_dir, set_weights

from flwr.common import logger, parameters_to_ndarrays
from flwr.common.typing import UserConfig
from flwr.server.strategy import FedAvg

PROJECT_NAME = "FLOWER-advanced-pytorch"


class CustomFedAvg(FedAvg):
    """A class that behaves like FedAvg but has extra functionality.

    This strategy: 
        (1) saves results to the filesystem, 
        (2) saves a checkpoint of the global  model when a new best is found, 
        (3) logs results to W&B if enabled.
    """

    def __init__(self, run_config: UserConfig, use_wandb: bool, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Create a directory where to save results from this run
        self.save_path, self.run_dir = create_run_dir(run_config)
        self.use_wandb = use_wandb
        # Initialise W&B if set
        if use_wandb:
            self._init_wandb_project()

        # Keep track of best acc
        self.best_acc_so_far = 0.0

        # A dictionary to store results as they come
        self.results = {}

    def _init_wandb_project(self):
        # init W&B
        wandb.init(project=PROJECT_NAME, name=f"{str(self.run_dir)}-ServerApp")

    def _store_results(self, tag: str, results_dict):
        """Store results in dictionary, then save as JSON."""
        # Update results dict
        if tag in self.results:
            self.results[tag].append(results_dict)
        else:
            self.results[tag] = [results_dict]

        # Save results to disk.
        # Note we overwrite the same file with each call to this function.
        # While this works, a more sophisticated approach is preferred
        # in situations where the contents to be saved are larger.
        with open(f"{self.save_path}/results.json", "w", encoding="utf-8") as fp:
            json.dump(self.results, fp)

    def _update_best_acc(self, round, accuracy, parameters):
        """Determines if a new best global model has been found.

        If so, the model checkpoint is saved to disk.
        """
        if accuracy > self.best_acc_so_far:
            self.best_acc_so_far = accuracy
            logger.log(INFO, "💡 New best global model found: %f", accuracy)
            # You could save the parameters object directly.
            # Instead we are going to apply them to a PyTorch
            # model and save the state dict.
            # Converts flwr.common.Parameters to ndarrays
            ndarrays = parameters_to_ndarrays(parameters)
            model = Net()
            set_weights(model, ndarrays)
            # Save the PyTorch model
            file_name = f"model_state_acc_{accuracy}_round_{round}.pth"
            torch.save(model.state_dict(), self.save_path / file_name)

    def store_results_and_log(self, server_round: int, tag: str, results_dict):
        """A helper method that stores results and logs them to W&B if enabled."""
        # Store results
        self._store_results(
            tag=tag,
            results_dict={"round": server_round, **results_dict},
        )

        if self.use_wandb:
            # Log centralized loss and metrics to W&B
            wandb.log(results_dict, step=server_round)

    def evaluate(self, server_round, parameters):
        """Run centralized evaluation if callback was passed to strategy init."""
        loss, metrics = super().evaluate(server_round, parameters)

        # Save model if new best central accuracy is found
        self._update_best_acc(server_round, metrics["centralized_accuracy"], parameters)

        # Store and log
        self.store_results_and_log(
            server_round=server_round,
            tag="centralized_evaluate",
            results_dict={"centralized_loss": loss, **metrics},
        )
        return loss, metrics

    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate results from federated evaluation."""
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)

        # Store and log
        self.store_results_and_log(
            server_round=server_round,
            tag="federated_evaluate",
            results_dict={"federated_evaluate_loss": loss, **metrics},
        )
        return loss, metrics

class FairFed(FedAvg):
    def __init__(self, *args, **kwargs):
        self.beta = kwargs.pop("beta", 0.1)
        self.ind_w = 0
        super().__init__(*args, **kwargs)
        
    def aggregate_evaluate(self, server_round, results, failures):
        """
        
        
        """
        indf_scores = [r.metrics["indf"] for _,r in results if 'indf' in r.metrics] 
        eod_scores = [r.metrics["eod"] for _,r in results if 'eod' in r.metrics]
        client_sizes = [r.num_examples for _,r in results]

        # global EOD
        avg_eod = sum(eod_scores) / len(eod_scores) if eod_scores else 0
        # global ind fairness
        avg_indf = sum(indf_scores) / len(indf_scores) if indf_scores else 0

        # glbal Acc.
        # TODO 
        # init adj
        adjusted_weights = []
        for indf, eod, num_examples in zip(indf_scores, eod_scores, client_sizes):
            # delta - group fairness
            dev_group = abs(eod - avg_eod)
            # delta - ind fairness
            dev_ind = abs(indf - avg_indf)
            # weighted delta 
            dev = self.ind_w*dev_ind + (1-self.ind_w)*dev_group
            # 
            adjusted_weight = max(1 - self.beta * dev, 0)
            # 
            adjusted_weights.append(adjusted_weight)

        total_weight = sum(adjusted_weights)
        norm_weights = [w / total_weight for w in adjusted_weights]

        aggregated_metrics = {
            'eod': sum(w * eod for w, eod in zip(norm_weights, eod_scores)),
            'indf': sum(w * indf for w, eod in zip(norm_weights, indf_scores)),            
        }
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        metrics.update(aggregated_metrics)
        print(f"Aggregated EOD: {aggregated_metrics['eod']}")
        print(f"Aggregated Ind Fairness: {aggregated_metrics['indf']}")        
        return loss, metrics
    