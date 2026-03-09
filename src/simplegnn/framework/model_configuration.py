"""
Model Configuration: Single configuration training and evaluation.

This module orchestrates the training and evaluation of a single GNN model
configuration. It handles the complete training pipeline including:
- Model initialization (from scratch or pretrained)
- Loss function and optimizer setup
- Learning rate scheduling
- Training loop execution (graph and node tasks)
- Validation and test set evaluation
- Early stopping and model pruning
- Results logging to CSV files
- Best model checkpointing

The ModelConfiguration class is the atomic execution unit called by
FrameworkMain for each hyperparameter configuration in a grid search.

Key Classes
-----------
EvaluationValues : Container for evaluation metrics
    Holds accuracy, loss, MAE, and ROC-AUC scores with standard deviations.
ModelConfiguration : Single configuration training executor
    Manages training loop, evaluation, and result saving for one
    hyperparameter configuration on one train/val/test split.

Usage Examples
--------------
Train a single configuration:

>>> config = ModelConfiguration(
...     run_id=0, k_val=0, graph_data=dataset,
...     model_data=(train_idx, val_idx, test_idx),
...     seed=42, para=parameters
... )
>>> config.train_configuration()

Evaluate on specific graphs:

>>> labels, predictions = config.evaluate_network(graph_ids=[0, 1, 2])

See Also
--------
framework.core.FrameworkMain : Main experiment orchestrator
models.model.GraphModel : PyTorch model class
"""
import datetime
import os
from typing import Tuple

import numpy as np
import pandas as pd
import sklearn
import torch
from torch import optim, nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from simplegnn.datasets.graph_dataset import GraphDataset, GraphData, CustomBatchLoader
from simplegnn.framework.utils.data_sampling import curriculum_sampling
from simplegnn.framework.utils.parameters import Parameters
from simplegnn.models.model import GraphModel
from simplegnn.utils.utils import get_k_lowest_nonzero_indices, valid_pruning_configuration, is_pruning
from simplegnn.utils.timer import TimeClass


class EvaluationValues:
    """
    Container for evaluation metrics during training and testing.

    Stores performance metrics computed during model evaluation including
    classification accuracy, regression MAE, loss values, and their standard
    deviations across validation runs.

    Attributes
    ----------
    accuracy : float
        Classification accuracy (0.0 to 1.0).
    accuracy_std : float
        Standard deviation of accuracy across runs.
    accuracy_roc_auc : float
        ROC-AUC score for classification tasks.
    loss : float
        Loss value (cross-entropy, MSE, etc.).
    loss_std : float
        Standard deviation of loss across runs.
    mae : float
        Mean Absolute Error for regression tasks.
    mae_std : float
        Standard deviation of MAE across runs.
    current_elements : int
        Number of elements evaluated (for averaging).
    """
    def __init__(self):
        """Initialize all metrics to zero."""
        self.accuracy = 0.0
        self.accuracy_std = 0.0
        self.accuracy_roc_auc = 0.0
        self.loss = 0.0
        self.loss_std = 0.0
        self.mae = 0.0
        self.mae_std = 0.0
        self.current_elements = 0



class ModelConfiguration:
    """
    Training executor for a single GNN configuration.

    Orchestrates the complete training and evaluation pipeline for one
    hyperparameter configuration on a specific train/validation/test split.
    This class is the atomic execution unit in the framework's grid search,
    called in parallel by FrameworkMain.run_configurations().

    Responsibilities:
    - Initialize GNN model (GraphModel) from configuration
    - Set up loss function, optimizer, and learning rate scheduler
    - Execute training loop with configurable batch sampling
    - Perform validation and test evaluation
    - Implement early stopping and model pruning
    - Log results to CSV files incrementally
    - Save best model checkpoints

    The class supports both graph-level tasks (classification, regression)
    and node-level tasks (node classification).

    Parameters
    ----------
    run_id : int
        Run identifier for random seed variation. Multiple runs with
        different seeds provide statistical robustness.
    k_val : int
        Validation fold index for k-fold cross-validation. Determines
        which subset is used for validation.
    graph_data : GraphDataset
        Complete dataset containing all graphs, node features, labels,
        and optional properties.
    model_data : Tuple[np.ndarray, np.ndarray, np.ndarray]
        Triple of (train_indices, val_indices, test_indices) specifying
        which graphs belong to each split.
    seed : int
        Random seed for reproducibility. Used for model weight
        initialization and batch sampling.
    para : Parameters
        Configuration container with hyperparameters, model architecture,
        paths, and task settings.

    Attributes
    ----------
    net : GraphModel
        Instantiated PyTorch model.
    criterion : torch.nn.Module
        Loss function (CrossEntropyLoss, MSELoss, etc.).
    optimizer : torch.optim.Optimizer
        Optimizer (Adam, SGD, AdamW, etc.).
    scheduler : torch.optim.lr_scheduler._LRScheduler or None
        Learning rate scheduler (StepLR, ReduceLROnPlateau, etc.).
    best_epoch : dict
        Dictionary tracking best epoch metrics (epoch, acc, loss, val_acc,
        val_loss, etc.) for model checkpointing.
    device : torch.device
        Execution device (cpu or cuda).
    dtype : torch.dtype
        Tensor precision (torch.float or torch.double).
    training_data : np.ndarray
        Graph indices for training set.
    validate_data : np.ndarray
        Graph indices for validation set.
    test_data : np.ndarray
        Graph indices for test set.
    class_weights : torch.Tensor or None
        Class weights for weighted loss (if enabled).
    results_path : Path
        Directory for saving results and models.

    Examples
    --------
    Standard training workflow:

    >>> config = ModelConfiguration(
    ...     run_id=0, k_val=0, graph_data=dataset,
    ...     model_data=(train_idx, val_idx, test_idx),
    ...     seed=42, para=parameters
    ... )
    >>> config.train_configuration()

    Transfer learning from pretrained model:

    >>> config.train_configuration(pretrained_network=pretrained_model)

    Notes
    -----
    **Training Loop:**

    For each epoch:
    1. Sample training batches (with optional curriculum sampling)
    2. Forward pass and loss computation
    3. Backward pass and weight update
    4. Validation evaluation (every validation_frequency epochs)
    5. Test evaluation (if test split exists)
    6. Early stopping check
    7. Model pruning (if configured)
    8. Learning rate scheduling
    9. CSV logging and model checkpointing

    **Result Files:**

    CSV results saved to:
    {results_path}/{dataset}/Results/{config_name}_Results_*.csv

    Best models saved to:
    {results_path}/{dataset}/Models/model_{config_name}_*.pt

    See Also
    --------
    models.model.GraphModel : PyTorch GNN model
    framework.core.FrameworkMain : Orchestrates multiple configurations
    """
    def __init__(self, run_id: int, k_val: int, graph_data: GraphDataset,
                 model_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
                 seed: int, para: Parameters):
        """
        Initialize ModelConfiguration with data splits and parameters.

        Parameters
        ----------
        run_id : int
            Run identifier for random seed variation.
        k_val : int
            Validation fold index.
        graph_data : GraphDataset
            Complete dataset.
        model_data : Tuple[np.ndarray, np.ndarray, np.ndarray]
            (train_indices, val_indices, test_indices).
        seed : int
            Random seed for reproducibility.
        para : Parameters
            Configuration with hyperparameters and model architecture.

        Notes
        -----
        Device and precision are configured from para.run_config.config.
        CUDA is used if available and specified in config, otherwise CPU.
        """
        self.num_epoch_samples = None
        self.best_epoch = None
        self.device = None
        self.dtype = None
        self.run_id = run_id
        self.k_val = k_val
        self.graph_data = graph_data
        self.training_data, self.validate_data, self.test_data = model_data
        self.seed = seed
        self.para = para
        self.results_path = para.run_config.config['paths']['results']
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.net = None
        self.class_weights = None
        self._csv_buffer = []
        self._csv_flush_interval = self.para.run_config.config.get('csv_flush_interval', 10)
        # get gpu or cpu: (cpu is recommended at the moment)
        if self.para.run_config.config.get('device', None) is not None:
            self.device = torch.device(self.para.run_config.config['device'] if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        self.dtype = torch.float
        if self.para.run_config.config.get('precision', 'float') == 'double':
            self.dtype = torch.double

    def train_configuration(self, pretrained_network=None):
        """
        Execute complete training and evaluation pipeline.

        Main entry point that orchestrates the full training process:
        model initialization, loss/optimizer setup, training loop execution,
        validation/test evaluation, early stopping, and result logging.

        This method runs for para.n_epochs epochs, evaluating on validation
        and test sets at configurable frequencies. Results are saved
        incrementally to CSV files, and the best model is checkpointed.

        Parameters
        ----------
        pretrained_network : torch.nn.Module or None, optional
            Pretrained model for transfer learning. If provided, its
            weights are used to initialize self.net instead of random
            initialization (default: None).

        Notes
        -----
        **Training Loop Structure:**

        For each epoch:
        1. Check early stopping criterion
        2. Generate training batches (with curriculum sampling if enabled)
        3. Compute class weights (if weighted_loss=True)
        4. Execute training step:
           - Graph tasks: train_graph_task()
           - Node tasks: train_node_task()
        5. Apply model pruning (if configured)
        6. Step learning rate scheduler
        7. Evaluate on validation set (every validation_frequency epochs)
        8. Evaluate on test set (every validation_frequency epochs)
        9. Save epoch results to CSV buffer
        10. Flush CSV buffer (every csv_flush_interval epochs)

        **Early Stopping:**

        Training stops early if best validation metric hasn't improved for
        para.early_stopping_patience epochs (if configured).

        **Best Model Tracking:**

        Tracks best epoch based on validation metrics:
        - Classification: Highest validation accuracy
        - Regression: Lowest validation MAE

        Best model weights are saved to:
        {results_path}/{dataset}/Models/model_{config_name}_*.pt

        **Skip Logic:**

        If results file already exists (from previous run), training is
        skipped to avoid redundant computation.

        Examples
        --------
        Train from scratch:

        >>> config.train_configuration()

        Transfer learning:

        >>> config.train_configuration(
        ...     pretrained_network=pretrained_model
        ... )

        See Also
        --------
        train_graph_task : Training step for graph-level tasks
        train_node_task : Training step for node-level tasks
        evaluate_results : Validation and test evaluation
        early_stopping : Early stopping criterion check
        """

        # Initialize the graph neural network
        self.initialize_model(pretrained_network=pretrained_network)
        # start the timer
        timer = TimeClass()
        # Define the loss function
        self.set_loss_function()
        # Define the optimizer
        self.set_optimizer()
        # Set up the file where the results are stored
        if not self.preprocess_writer():
            # Run already exists, so we do not run the training again
            print(f"Run {self.run_id} already exists, skipping training.")
            return
        # Define the scheduler
        self.set_scheduler()
        # Initialize the best epoch
        self.best_epoch = {"epoch": 0, "acc": 0.0, "roc_auc": 0.0, "loss": 1000000.0, "val_acc": 0.0,  "val_roc_auc": 0.0, "val_loss": 1000000.0, "val_mae": 1000000.0}
        # Create the seeds for the different epochs and validation runs
        seeds = np.arange(self.para.n_epochs*self.para.n_val_runs)
        seeds = np.reshape(seeds, (self.para.n_epochs, self.para.n_val_runs))
        # set data to device
        #self.graph_data.to(self.device)

        # Run through the epochs
        for epoch in range(self.para.n_epochs):
            # Test early stopping criterion
            if self.early_stopping(epoch):
                print(f"Early stopping at epoch {epoch}")
                self._flush_csv_buffer()
                break

            timer.measure("epoch")
            self.net.epoch = epoch
            epoch_values = EvaluationValues()
            validation_values = EvaluationValues()
            test_values = EvaluationValues()
            train_batches = self.get_train_batches(seeds, epoch)

            # if weighted_loss is set to true, get the class weights
            if self.para.run_config.config.get('weighted_loss', False):
                # get class counts per batch
                self.class_weights = torch.zeros((len(train_batches), self.graph_data.num_classes), dtype=self.dtype)
                for i in range(len(train_batches)):
                    self.class_weights[i] = torch.unique(self.graph_data.y[train_batches[i]], return_counts=True)[1]
                self.class_weights = 1.0 - torch.einsum('ij,i->ij', self.class_weights, 1.0/self.class_weights.sum(dim=1))



            self.num_epoch_samples = sum([batch.size for batch in train_batches])
            self.net.train(True)
            if self.para.run_config.config['task'] in ['graph_regression', 'graph_classification']:
                self.train_graph_task(epoch=epoch, values=(epoch_values, validation_values, test_values), train_batches=train_batches, timer=timer)
            elif self.para.run_config.config['task'] == 'node_classification':
                self.train_node_task(epoch=epoch, values=(epoch_values, validation_values, test_values), train_batches=train_batches, timer=timer)


            # TODO Pruning
            if valid_pruning_configuration(self.para, epoch):
                self.model_pruning(epoch)

            # Step the scheduler
            if self.scheduler is not None:
                if self.para.run_config.config['scheduler']['type'] == 'ReduceLROnPlateau':
                    self.scheduler.step(validation_values.loss)
                else:
                    self.scheduler.step()

            # Evaluate the results on training, validation and test set (only if specified in the config or for evaluation)
            validation_frequency = self.para.run_config.config.get('validation_frequency', 1)
            is_validation_epoch = (epoch + 1) % validation_frequency == 0 or epoch == self.para.n_epochs - 1
            if is_validation_epoch and self.validate_data.size != 0:
                epoch_values, validation_values, test_values = self.evaluate_results(epoch=epoch,train_values=epoch_values, validation_values=validation_values, test_values=test_values, evaluation_type='validation')
            # check wheter there is a test split
            if is_validation_epoch and self.test_data.size != 0:
                epoch_values, validation_values, test_values = self.evaluate_results(epoch=epoch,train_values=epoch_values, validation_values=validation_values, test_values=test_values, evaluation_type='test')

            timer.measure("epoch")
            epoch_time = timer.get_flag_time("epoch")

            # Write the results to the results file
            self.postprocess_writer(epoch, epoch_time, epoch_values, validation_values, test_values)


    def evaluate_network(self, graph_ids, do_print=False, with_loss=False):
        """
        Evaluate model on specified graphs.

        Runs the model in evaluation mode on a given set of graph indices
        and computes predictions. Supports both graph-level tasks
        (classification, regression) and node-level tasks.

        Parameters
        ----------
        graph_ids : np.ndarray or list
            Indices of graphs to evaluate from self.graph_data.
        do_print : bool, optional
            Print evaluation metrics to console (default: False).
        with_loss : bool, optional
            Compute and print loss value when do_print=True
            (default: False).

        Returns
        -------
        target_values : torch.Tensor
            True labels/values for the evaluated graphs.
            - Graph classification: Shape (num_graphs,)
            - Graph regression: Shape (num_graphs, output_dim)
            - Node classification: Shape (num_nodes,)
        target_outputs : torch.Tensor
            Model predictions for the evaluated graphs.
            - Graph classification: Shape (num_graphs, num_classes)
            - Graph regression: Shape (num_graphs, output_dim)
            - Node classification: Shape (num_nodes, num_classes)

        Notes
        -----
        Model is automatically set to eval() mode. Evaluation uses
        torch.no_grad() for efficiency (implemented in task-specific
        methods).

        **Printed Metrics (if do_print=True):**

        - Graph classification: Accuracy and optionally loss
        - Graph regression: Mean Absolute Error (MAE) and RMSE

        See Also
        --------
        evaluate_graph_task : Graph-level evaluation implementation
        evaluate_node_task : Node-level evaluation implementation
        """
        # set data to device
        #self.graph_data.to(self.device)
        self.net.eval()
        # Evaluate the network on the given graph ids

        if self.para.run_config.task in ['graph_regression', 'graph_classification']:
            target_values, target_outputs = self.evaluate_graph_task(graph_ids)
            # print the accuracy
            if do_print and self.para.run_config.task == 'graph_classification':
                predictions = torch.argmax(target_outputs, dim=1)
                accuracy = 100 * torch.sum(predictions == target_values).item() / len(target_values)
                if with_loss:
                    self.set_loss_function()
                    loss = self.criterion(target_outputs, target_values).item()
                print(f"Accuracy: {accuracy} %, Loss: {loss}")
            else:
                if do_print:
                    print(f"Evaluation completed for graph regression task.")
                    mae_error = torch.mean(torch.abs(target_values - target_outputs))
                    rsme_error = torch.mean(torch.sqrt((target_values - target_outputs) ** 2))
                    print(f"Mean Absolute Error: {mae_error}")
        elif self.para.run_config.task == 'node_classification':
            target_values, target_outputs = self.evaluate_node_task(graph_ids)
        else:
            raise ValueError(f"Task {self.para.run_config.task} not implemented")

        return target_values, target_outputs


    def initialize_model(self, pretrained_network):
        """
        Initialize GNN model from scratch or pretrained weights.

        Creates a GraphModel instance with the specified architecture from
        para.layers. If a pretrained network is provided, loads its weights
        for transfer learning.

        Parameters
        ----------
        pretrained_network : torch.nn.Module or None
            Pretrained model for transfer learning. If None, initializes
            model with random weights using the seed.

        Notes
        -----
        Model is moved to self.device after initialization. If
        pretrained_network is provided, its state_dict is loaded into the
        new model using torch.load_state_dict().

        See Also
        --------
        models.model.GraphModel : Model class
        """
        print(f'Initializing network with seed {self.seed}')
        if pretrained_network is not None:
            print('Using pretrained network')
            self.net = pretrained_network
        else:
            self.net = GraphModel(graph_data=self.graph_data, para=self.para, seed=self.seed, device=self.device)

        # set the network to device
        self.net.to(self.device)
        print(f'Network initialized with seed {self.seed}')



    def set_loss_function(self, *args, **kwargs):
        """
        Configure loss function based on task type.

        Sets self.criterion to the appropriate PyTorch loss function:
        - Graph classification: CrossEntropyLoss
        - Graph regression: MSELoss or MAELoss
        - Node classification: CrossEntropyLoss

        Notes
        -----
        Loss function is selected from para.run_config.config['loss'].
        Weighted loss is supported for classification tasks if
        config['weighted_loss']=True (weights computed per batch).
        """
        if self.para.run_config.loss == 'CrossEntropyLoss':
            self.criterion = nn.CrossEntropyLoss(*args, **kwargs)
        elif self.para.run_config.loss in ['MeanSquaredError', 'MSELoss', 'mse', 'MSE']:
            self.criterion = nn.MSELoss(*args, **kwargs)
        elif self.para.run_config.loss in ['RootedMeanSquaredError', 'RMSELoss', 'rmse', 'RMSE']:
            def RSMELoss(input, target):
                return torch.sqrt(F.mse_loss(input, target) + 1e-8)
            self.criterion = RSMELoss
        elif self.para.run_config.loss in ['L1Loss', 'l1', 'L1', 'mean_absolute_error', 'mae', 'MAE', 'MeanAbsoluteError']:
            self.criterion = nn.L1Loss(*args, **kwargs)
        elif self.para.run_config.loss in ['BCELoss', 'bce', 'BCE']:
            self.criterion = nn.BCELoss(*args, **kwargs)
        elif self.para.run_config.loss in ['BCEWithLogitsLoss', 'bce_with_logits', 'BCEWithLogits']:
            self.criterion = nn.BCEWithLogitsLoss(*args, **kwargs)
        elif self.para.run_config.loss in ['NLLLoss', 'nll', 'NLL']:
            self.criterion = nn.NLLLoss(*args, **kwargs)
        else:
            raise ValueError(f"Loss function {self.para.run_config.loss} not implemented")

    def set_optimizer(self):
        """
        Configure optimizer from configuration.

        Sets self.optimizer to the specified optimizer (Adam, SGD, AdamW,
        etc.) with learning rate and weight decay from configuration.

        Notes
        -----
        Optimizer type and hyperparameters are read from
        para.run_config.config['optimizer'].

        Supported optimizers: Adam, SGD, AdamW, RMSprop, etc.
        """
        if self.para.run_config.optimizer == 'Adam':
            opt = optim.Adam
        elif self.para.run_config.optimizer == 'AdamW':
            opt = optim.AdamW
        elif self.para.run_config.optimizer == 'SGD':
            opt = optim.SGD
        elif self.para.run_config.optimizer == 'RMSprop':
            opt = optim.RMSprop
        elif self.para.run_config.optimizer == 'Adadelta':
            opt = optim.Adadelta
        elif self.para.run_config.optimizer == 'Adagrad':
            opt = optim.Adagrad
        else:
            opt = optim.Adam

        self.optimizer = opt(self.net.parameters(), lr=self.para.learning_rate, weight_decay=self.para.run_config.weight_decay)

    def set_scheduler(self):
        """
        Configure learning rate scheduler from configuration.

        Sets self.scheduler to the specified LR scheduler (StepLR,
        ReduceLROnPlateau, etc.) if configured, otherwise None.

        Notes
        -----
        Scheduler type and parameters are read from
        para.run_config.config['scheduler'] if present.

        Supported schedulers: StepLR, ReduceLROnPlateau, etc.

        ReduceLROnPlateau requires validation loss as input during
        scheduler.step() calls.
        """
        """
        Variable learning rate
        """
        if self.para.run_config.config.get('scheduler', None) is not None:
            scheduler = self.para.run_config.config.get('scheduler')
            if isinstance(scheduler, bool):
                if scheduler is True:
                    raise ValueError("Scheduler is set to True, but no scheduler is defined")
                else:
                    self.scheduler = None
                    return
            scheduler_type = scheduler.get('type', None)
            if scheduler_type == 'StepLR':
                self.scheduler = StepLR(self.optimizer, step_size=scheduler.get('step_size', None), gamma=scheduler.get('gamma', None))
            elif scheduler_type == 'ReduceLROnPlateau':
                self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=scheduler.get('patience', 10), min_lr=scheduler.get('min_lr', 0), factor=scheduler.get('factor', 0.1))


    def early_stopping(self, epoch):
        if self.para.run_config.config.get('early_stopping', {'enabled': False})['enabled']:
            if epoch - self.best_epoch["epoch"] > self.para.run_config.config['early_stopping']['patience']:
                if self.para.print_results:
                    print(f"Early stopping at epoch {epoch}")
                return True
        return False

    def test_weight_update(self, weights):
        weight_changes = []
        for i, layer in enumerate(self.net.net_layers):
            change = np.array(
                [weights[i][j] - x.item() for j, x in enumerate(layer.Param_W)]).flatten().reshape(1, -1)
            weight_changes.append(change)
            # save to three differen csv files using pandas
            # df = pd.DataFrame(change)
            # df.to_csv(f'Results/Parameter/layer_{i}_change.csv', header=False, index=False, mode='a')
            # if there is some change print that the layer trains
            if np.count_nonzero(change) > 0:
                print(f'Layer {i} has updated')
            else:
                print(f'Layer {i} has not updated')

    def model_pruning(self, epoch):
        # prune each five epochs

        print('Pruning')
        # iterate over the layers of the neural net
        for i, layer in enumerate(self.net.net_layers):
            pruning_per_layer = self.para.run_config.config['prune']['percentage'][i]
            # use total number of epochs, the epoch step and the pruning percentage
            pruning_per_layer /= (self.para.n_epochs / self.para.run_config.config['prune']['epochs']) - 1

            # get tensor from the parameter_list layer.Param_W
            layer_tensor = torch.abs(torch.tensor(layer.Param_W) * torch.tensor(layer.mask))
            # print number of non zero entries in layer_tensor
            print(f'Number of non zero entries in before pruning {layer.name}: {torch.count_nonzero(layer_tensor)}')
            # get the indices of the trainable parameters with lowest absolute max(1, 1%)
            k = int(layer_tensor.size(0) * pruning_per_layer)
            if k != 0:
                low = torch.topk(layer_tensor, k, largest=False)
                lowest_indices = get_k_lowest_nonzero_indices(layer_tensor, k)
                # set all indices in layer.mask to zero
                layer.mask[lowest_indices] = 0
                layer.Param_W.data = layer.Param_W_original * layer.mask
                # for c, graph_weight_distribution in enumerate(layer.weight_distribution):
                #     new_graph_weight_distribution = None
                #     for [i, j, pos] in graph_weight_distribution:
                #         # if pos is in lowest_indices do nothing else append to new_graph_weight_distribution
                #         if pos in lowest_indices:
                #             pass
                #         else:
                #             if new_graph_weight_distribution is None:
                #                 new_graph_weight_distribution = np.array([i, j, pos])
                #             else:
                #                 # add [i, j, pos] to new_graph_weight_distribution
                #                 new_graph_weight_distribution = np.vstack((new_graph_weight_distribution, [i, j, pos]))
                #     layer.weight_distribution[c] = new_graph_weight_distribution

            # print number of non zero entries in layer.Param_W
            print(
                f'Number of non zero entries in layer after pruning {layer.name}: {torch.count_nonzero(layer.Param_W)}')
        if is_pruning(self.para.run_config.config):
            for i, layer in enumerate(self.net.net_layers):
                # get tensor from the parameter_list layer.Param_W
                layer_tensor = torch.abs(torch.tensor(layer.Param_W).clone().detach() * torch.tensor(layer.mask))
                # print number of non zero entries in layer_tensor
                print(
                    f'Number of non zero entries in layer {layer.name}: {torch.count_nonzero(layer_tensor)}/{torch.numel(layer_tensor)}')

                # multiply the Param_W with the mask
                layer.Param_W.data = layer.Param_W.data * layer.mask


    def evaluate_results(self, epoch: int,
                         train_values: EvaluationValues,
                         validation_values: EvaluationValues,
                         test_values: EvaluationValues,
                         evaluation_type,
                         outputs=None,
                         labels=None,
                         batch_idx=0,
                         batch_length=0,
                         num_batches=0,
                         batches=None):
        """
        Evaluate model performance and update metrics.

        Comprehensive evaluation method that computes metrics (accuracy,
        loss, MAE, ROC-AUC) for training, validation, or test sets.
        Updates EvaluationValues objects with running averages and handles
        best model checkpointing.

        Parameters
        ----------
        epoch : int
            Current training epoch.
        train_values : EvaluationValues
            Container for training set metrics.
        validation_values : EvaluationValues
            Container for validation set metrics.
        test_values : EvaluationValues
            Container for test set metrics.
        evaluation_type : str
            Type of evaluation: 'training', 'validation', or 'test'.
        outputs : torch.Tensor or None, optional
            Model predictions for training evaluation (default: None).
            Required when evaluation_type='training'.
        labels : torch.Tensor or None, optional
            True labels for training evaluation (default: None).
            Required when evaluation_type='training'.
        batch_idx : int, optional
            Current batch index for progress printing (default: 0).
        batch_length : int, optional
            Number of samples in current batch (default: 0).
        num_batches : int, optional
            Total number of batches in epoch (default: 0).
        batches : list or None, optional
            List of batches (default: None).

        Returns
        -------
        Tuple[EvaluationValues, EvaluationValues, EvaluationValues]
            Updated (train_values, validation_values, test_values).

        Notes
        -----
        **Evaluation Types:**

        - training: Updates running metrics during training epoch using
          provided outputs and labels. Computes batch metrics and
          maintains running average.

        - validation: Evaluates entire validation set, computes metrics,
          and checks if this is the best epoch for model checkpointing.

        - test: Evaluates entire test set and computes final metrics.

        **Metrics Computed:**

        Classification tasks:
        - Accuracy: Percentage of correct predictions
        - ROC-AUC: Area under ROC curve (if configured)
        - Loss: Cross-entropy loss

        Regression tasks:
        - MAE: Mean Absolute Error
        - MAE_std: Standard deviation of absolute errors
        - Loss: MSE loss

        **Best Model Tracking:**

        For validation evaluation, compares current metrics against
        self.best_epoch and saves model checkpoint if improved:
        - Classification: Best validation accuracy
        - Regression: Best (lowest) validation MAE

        Model saved to:
        {results_path}/{dataset}/Models/model_{config_name}_*.pt

        **Output Transformations:**

        For regression tasks, supports inverse transformations to
        original scale using config['invert_outputs']:
        - standard: Standardization (z-score)
        - minmax: Min-max normalization
        - minmax_zero: Min-max to [-1, 1]

        See Also
        --------
        evaluate_graph_task : Graph-level evaluation
        evaluate_node_task : Node-level evaluation
        EvaluationValues : Metrics container
        """
        self.net.eval()
        if evaluation_type == 'training':
            batch_acc = 0
            # if num classes is one calculate the mae and mae_std or if the task is regression
            if self.para.run_config.task == 'graph_regression':
                # flatten the labels and outputs
                flatten_labels = labels.detach().clone().flatten()
                flatten_outputs = outputs.detach().clone().flatten()
                if self.para.run_config.config.get('output_features_inverse', None) is not None:
                    flatten_labels = GraphData.transform_data(flatten_labels, self.para.run_config.config['output_features_inverse'])
                    flatten_outputs = GraphData.transform_data(flatten_outputs, self.para.run_config.config['output_features_inverse'])
                if self.para.run_config.config.get('invert_outputs', None) is not None:
                    if isinstance(self.para.run_config.config['invert_outputs'], dict):
                        if self.para.run_config.config['invert_outputs'].get('normalization', 'standard') == 'standard':
                            flatten_labels = flatten_labels*(self.graph_data.data['original_y'].std() + 1e-8) + self.graph_data.data['original_y'].mean()
                            flatten_outputs = flatten_outputs*(self.graph_data.data['original_y'].std() + 1e-8) + self.graph_data.data['original_y'].mean()
                        elif self.para.run_config.config['invert_outputs'].get('normalization', 'standard') == 'minmax':
                            flatten_labels = flatten_labels * (self.graph_data.data['original_y'].max() - self.graph_data.data['original_y'].min() + 1e-8) + self.graph_data.data['original_y'].min()
                            flatten_outputs = flatten_outputs * (self.graph_data.data['original_y'].max() - self.graph_data.data['original_y'].min() + 1e-8) + self.graph_data.data['original_y'].min()
                        elif self.para.run_config.config['invert_outputs'].get('normalization', 'standard') == 'minmax_zero':
                            flatten_labels = (0.5 * flatten_labels + 0.5) * (self.graph_data.data['original_y'].max() - self.graph_data.data['original_y'].min() + 1e-8) + self.graph_data.data['original_y'].min()


                batch_mae = torch.mean(torch.abs(flatten_labels - flatten_outputs))
                batch_mae_std = torch.std(torch.abs(flatten_labels - flatten_outputs))
                train_values.mae = (train_values.mae * train_values.current_elements + batch_mae * batch_length) / (train_values.current_elements + batch_length)
                train_values.mae_std = (train_values.mae_std * train_values.current_elements + batch_mae_std * batch_length) / (train_values.current_elements + batch_length)
            else:
                prediction = torch.argmax(outputs, dim=1)
                batch_acc = 100 * torch.sum(prediction == labels).item() / len(labels)
                # accuracy
                train_values.accuracy = (train_values.accuracy * train_values.current_elements + batch_acc * batch_length) / (train_values.current_elements + batch_length)
                # roc_auc
                # if undersampling is used, the batch always contains all classes otherwise roc_auc cannot be calculated
                if self.para.run_config.config.get('training_data_sampling', None) is not None and self.para.run_config.config['training_data_sampling'].get('type', None) == 'undersampling':
                    if self.para.run_config.config.get('evaluation_metric', 'accuracy') == 'roc_auc':
                        batch_roc_auc = sklearn.metrics.roc_auc_score(labels, prediction)
                        train_values.accuracy_roc_auc = (train_values.accuracy_roc_auc * train_values.current_elements + batch_roc_auc * batch_length) / (train_values.current_elements + batch_length)
            train_values.current_elements += batch_length
            if self.para.print_results:
                if self.graph_data.num_classes == 1 or self.para.run_config.task == 'graph_regression':
                    print(
                        "\tepoch: {}/{}, batch: {}/{}, loss: {}, acc: {} %, mae: {}, mae_std: {}".format(epoch + 1,
                                                                                                         self.para.n_epochs,
                                                                                                         batch_idx + 1,
                                                                                                         num_batches,
                                                                                                         train_values.loss,
                                                                                                         batch_acc,
                                                                                                         train_values.mae,
                                                                                                         train_values.mae_std))
                else:
                    print("\tepoch: {}/{}, batch: {}/{}, loss: {}, acc: {} % ".format(epoch + 1, self.para.n_epochs,
                                                                                      batch_idx + 1,
                                                                                      num_batches,
                                                                                      train_values.loss, batch_acc))
            self.para.count += 1

            if self.para.save_prediction_values:
                # print outputs and labels to a csv file
                outputs_np = outputs.detach().numpy()
                # transpose the numpy array
                outputs_np = outputs_np.T
                df = pd.DataFrame(outputs_np)
                # show only two decimal places
                df = df.round(2)
                df.to_csv("Results/Parameter/training_predictions.csv", header=False, index=False, mode='a')
                labels_np = labels.detach().numpy()
                labels_np = labels_np.T
                df = pd.DataFrame(labels_np)
                df.to_csv("Results/Parameter/training_predictions.csv", header=False, index=False, mode='a')

        elif evaluation_type == 'validation':
            '''
            Evaluate the validation accuracy for each epoch
            '''
            if self.validate_data.size != 0:
                if self.para.run_config.task in ['graph_classification', 'graph_regression']:
                    labels, outputs = self.evaluate_graph_task(self.validate_data)
                    # check if output is two dimensional and task is graph classification
                    if self.para.run_config.config.get('task', None) == 'graph_classification' and len(outputs.shape) > 1 and outputs.shape[1] != 1:
                        labels = torch.nn.functional.one_hot(labels, num_classes=self.graph_data.num_classes).to(self.dtype).to(self.device)
                    elif self.para.run_config.config.get('task', None) == 'graph_regression' and len(outputs.shape) > 1 and outputs.shape[1] == 1:
                        labels = labels.unsqueeze(1)
                elif self.para.run_config.task == 'node_classification':
                    labels, outputs = self.evaluate_node_task(self.validate_data)
                else:
                    raise ValueError(f"Task {self.para.run_config.task} not implemented")
                # get validation loss
                validation_loss = self.criterion(outputs, labels).item()
                validation_values.loss = validation_loss
                if self.para.run_config.task == 'graph_regression':
                    flatten_labels = labels.detach().clone().flatten()
                    flatten_outputs = outputs.detach().clone().flatten()
                    if self.para.run_config.config.get('output_features_inverse', None) is not None:
                        flatten_labels = GraphData.transform_data(flatten_labels, self.para.run_config.config[
                            'output_features_inverse'])
                        flatten_outputs = GraphData.transform_data(flatten_outputs, self.para.run_config.config[
                            'output_features_inverse'])
                    if self.para.run_config.config.get('invert_outputs', None) is not None:
                        if isinstance(self.para.run_config.config['invert_outputs'], dict):
                            if self.para.run_config.config['invert_outputs'].get('normalization',
                                                                                 'standard') == 'standard':
                                flatten_labels = flatten_labels * (self.graph_data.data['original_y'].std() + 1e-8) + \
                                                 self.graph_data.data['original_y'].mean()
                                flatten_outputs = flatten_outputs * (self.graph_data.data['original_y'].std() + 1e-8) + \
                                                  self.graph_data.data['original_y'].mean()
                            elif self.para.run_config.config['invert_outputs'].get('normalization',
                                                                                   'standard') == 'minmax':
                                flatten_labels = flatten_labels * (
                                            self.graph_data.data['original_y'].max() - self.graph_data.data[
                                        'original_y'].min() + 1e-8) + self.graph_data.data['original_y'].min()
                                flatten_outputs = flatten_outputs * (
                                            self.graph_data.data['original_y'].max() - self.graph_data.data[
                                        'original_y'].min() + 1e-8) + self.graph_data.data['original_y'].min()
                            elif self.para.run_config.config['invert_outputs'].get('normalization',
                                                                                   'standard') == 'minmax_zero':
                                flatten_labels = (0.5 * flatten_labels + 0.5) * (
                                            self.graph_data.data['original_y'].max() - self.graph_data.data[
                                        'original_y'].min() + 1e-8) + self.graph_data.data['original_y'].min()
                    validation_mae = torch.mean(torch.abs(flatten_labels - flatten_outputs))
                    validation_values.mae = validation_mae
                    validation_mae_std = torch.std(torch.abs(flatten_labels - flatten_outputs))
                    validation_values.mae_std = validation_mae_std
                else:
                    prediction = torch.argmax(outputs, dim=1)
                    if len(labels.shape) > 1:
                        labels = torch.argmax(labels, dim=1)
                    validation_acc = 100 * torch.sum(prediction==labels).item() / len(labels)
                    validation_values.accuracy = validation_acc
                    if self.para.run_config.config.get('evaluation_metric', 'accuracy') == 'roc_auc':
                        # roc_auc
                        validation_roc_auc = sklearn.metrics.roc_auc_score(labels, prediction)
                        validation_values.accuracy_roc_auc = validation_roc_auc

                # update best epoch
                if self.para.run_config.task == 'graph_regression':
                    if validation_values.mae <= self.best_epoch["val_mae"] or valid_pruning_configuration(self.para, epoch):
                        self.best_epoch["epoch"] = epoch
                        self.best_epoch["acc"] = train_values.accuracy
                        self.best_epoch["roc_auc"] = train_values.accuracy_roc_auc
                        self.best_epoch["loss"] = train_values.loss
                        self.best_epoch["val_acc"] = validation_values.accuracy
                        self.best_epoch["val_roc_auc"] = validation_values.accuracy_roc_auc
                        self.best_epoch["val_loss"] = validation_values.loss
                        self.best_epoch["val_mae"] = validation_values.mae
                        self.best_epoch["val_mae_std"] = validation_values.mae_std
                        # save the best model
                        best_model_path = self.results_path.joinpath(f'{self.para.db}/Models/')
                        if not os.path.exists(best_model_path):
                            os.makedirs(best_model_path)
                        # Save the model if best model is used
                        if 'best_model' in self.para.run_config.config and self.para.run_config.config['best_model']:
                            final_path = self.results_path.joinpath(f'{self.para.db}/Models/model_{self.para.config_id}_run_{self.run_id}_val_step_{self.k_val}.pt')
                            torch.save(self.net.state_dict(),final_path)


                else:
                    acc_condition = (validation_values.accuracy > self.best_epoch["val_acc"] or (validation_values.accuracy == self.best_epoch["val_acc"] and validation_loss < self.best_epoch["val_loss"]))
                    roc_condition = (validation_values.accuracy_roc_auc > self.best_epoch["val_roc_auc"] or (validation_values.accuracy_roc_auc == self.best_epoch["val_roc_auc"] and validation_loss < self.best_epoch["val_loss"]))
                    loss_condition = (validation_loss < self.best_epoch["val_loss"])
                    condition = False
                    if self.para.run_config.config.get('evaluation_metric', 'accuracy') == 'accuracy':
                        condition = acc_condition
                    elif self.para.run_config.config.get('evaluation_metric', 'accuracy') == 'roc_auc':
                        condition = roc_condition
                    elif self.para.run_config.config.get('evaluation_metric', 'accuracy') == 'loss':
                        condition = loss_condition
                    # check if pruning is on, then save the best model in the last pruning epoch
                    if condition or valid_pruning_configuration(self.para, epoch):
                        self.best_epoch["epoch"] = epoch
                        self.best_epoch["acc"] = train_values.accuracy
                        self.best_epoch["roc_auc"] = train_values.accuracy_roc_auc
                        self.best_epoch["loss"] = train_values.loss
                        self.best_epoch["val_acc"] = validation_values.accuracy
                        self.best_epoch["val_roc_auc"] = validation_values.accuracy_roc_auc
                        self.best_epoch["val_loss"] = validation_values.loss
                        # save the best model
                        best_model_path = self.results_path.joinpath(f'{self.para.db}/Models/')
                        if not os.path.exists(best_model_path):
                            os.makedirs(best_model_path)
                        # Save the model if best model is used
                        if self.para.run_config.config.get('best_model', False) or self.para.run_config.config.get('save_best_model', False):
                            final_path = self.results_path.joinpath(f'{self.para.db}/Models/model_{self.para.config_id}_run_{self.run_id}_val_step_{self.k_val}.pt')
                            torch.save(self.net.state_dict(), final_path)

            if self.para.save_prediction_values:
                # print outputs and labels to a csv file
                outputs_np = outputs.detach().numpy()
                # transpose the numpy array
                outputs_np = outputs_np.T
                df = pd.DataFrame(outputs_np)
                # show only two decimal places
                df = df.round(2)
                df.to_csv("Results/Parameter/validation_predictions.csv", header=False, index=False, mode='a')
                labels_np = labels.detach().numpy()
                labels_np = labels_np.T
                df = pd.DataFrame(labels_np)
                df.to_csv("Results/Parameter/validation_predictions.csv", header=False, index=False, mode='a')

        elif evaluation_type == 'test':
            # Test accuracy
            # print only if run best model is used
            if self.para.run_config.config.get('best_model', False):
                if self.para.run_config.task in ['graph_classification', 'graph_regression']:
                    labels, outputs = self.evaluate_graph_task(self.test_data)
                    # check if output is two dimensional and task is graph classification
                    if self.para.run_config.config.get('task', None) == 'graph_classification' and len(outputs.shape) > 1 and outputs.shape[1] != 1:
                        labels = torch.nn.functional.one_hot(labels, num_classes=self.graph_data.num_classes).to(self.dtype).to(self.device)
                    elif self.para.run_config.config.get('task', None) == 'graph_regression' and len(outputs.shape) > 1 and outputs.shape[1] == 1:
                        labels = labels.unsqueeze(1)
                elif self.para.run_config.task == 'node_classification':
                    labels, outputs = self.evaluate_node_task(self.test_data)
                else:
                    raise ValueError(f"Task {self.para.run_config.task} not implemented")

                test_loss = self.criterion(outputs, labels).item()
                test_values.loss = test_loss
                if self.para.run_config.task == 'graph_regression':
                    flatten_labels = labels.detach().clone().flatten()
                    flatten_outputs = outputs.detach().clone().flatten()
                    if self.para.run_config.config.get('output_features_inverse', None) is not None:
                        flatten_labels = GraphData.transform_data(flatten_labels, self.para.run_config.config[
                            'output_features_inverse'])
                        flatten_outputs = GraphData.transform_data(flatten_outputs, self.para.run_config.config[
                            'output_features_inverse'])
                    if self.para.run_config.config.get('invert_outputs', None) is not None:
                        if isinstance(self.para.run_config.config['invert_outputs'], dict):
                            if self.para.run_config.config['invert_outputs'].get('normalization',
                                                                                 'standard') == 'standard':
                                flatten_labels = flatten_labels * (self.graph_data.data['original_y'].std() + 1e-8) + \
                                                 self.graph_data.data['original_y'].mean()
                                flatten_outputs = flatten_outputs * (self.graph_data.data['original_y'].std() + 1e-8) + \
                                                  self.graph_data.data['original_y'].mean()
                            elif self.para.run_config.config['invert_outputs'].get('normalization',
                                                                                   'standard') == 'minmax':
                                flatten_labels = flatten_labels * (
                                            self.graph_data.data['original_y'].max() - self.graph_data.data[
                                        'original_y'].min() + 1e-8) + self.graph_data.data['original_y'].min()
                                flatten_outputs = flatten_outputs * (
                                            self.graph_data.data['original_y'].max() - self.graph_data.data[
                                        'original_y'].min() + 1e-8) + self.graph_data.data['original_y'].min()
                            elif self.para.run_config.config['invert_outputs'].get('normalization',
                                                                                   'standard') == 'minmax_zero':
                                flatten_labels = (0.5 * flatten_labels + 0.5) * (
                                            self.graph_data.data['original_y'].max() - self.graph_data.data[
                                        'original_y'].min() + 1e-8) + self.graph_data.data['original_y'].min()
                    test_mae = torch.mean(torch.abs(flatten_labels - flatten_outputs))
                    test_values.mae = test_mae
                    test_mae_std = torch.std(torch.abs(flatten_labels - flatten_outputs))
                    test_values.mae_std = test_mae_std
                else:
                    prediction = torch.argmax(outputs, dim=1)
                    if len(labels.shape) > 1:
                        labels = torch.argmax(labels, dim=1)
                    test_acc = 100 * torch.sum(prediction == labels).item() / len(labels)
                    test_values.accuracy = test_acc
                    if self.para.run_config.config.get('evaluation_metric', 'accuracy') == 'roc_auc':
                        # roc_auc
                        test_roc_auc = sklearn.metrics.roc_auc_score(labels, prediction)
                        test_values.accuracy_roc_auc = test_roc_auc

                if self.para.print_results:
                    np_labels = labels.detach().numpy()
                    np_outputs = outputs.detach().numpy()
                    # np array of correct/incorrect predictions
                    labels_argmax = np_labels.argmax(axis=1)
                    outputs_argmax = np_outputs.argmax(axis=1)
                    # change if task is graph_regression
                    if 'task' in self.para.run_config.config and self.para.run_config.config['task'] == 'graph_regression':
                        np_correct = np_labels - np_outputs
                    else:
                        np_correct = labels_argmax == outputs_argmax
                    # print entries of np_labels and np_outputs
                    for j, data_pos in enumerate(self.test_data, 0):
                        print(data_pos, np_labels[j], np_outputs[j], np_correct[j])

                if self.para.save_prediction_values:
                    # print outputs and labels to a csv file
                    outputs_np = outputs.detach().numpy()
                    # transpose the numpy array
                    outputs_np = outputs_np.T
                    df = pd.DataFrame(outputs_np)
                    # show only two decimal places
                    df = df.round(2)
                    df.to_csv("Results/Parameter/test_predictions.csv", header=False, index=False, mode='a')
                    labels_np = labels.detach().numpy()
                    labels_np = labels_np.T
                    df = pd.DataFrame(labels_np)
                    df.to_csv("Results/Parameter/test_predictions.csv", header=False, index=False, mode='a')
        self.net.train()
        return train_values, validation_values, test_values

    def preprocess_writer(self)-> bool:
        if self.run_id == 0 and self.k_val == 0:
            # create a file about the net details including (net, optimizer, learning rate, loss function, batch size, number of classes, number of epochs, balanced data, dropout)
            file_name = f'{self.para.db}_{self.para.config_id}_Network.txt'
            final_path = self.results_path.joinpath(f'{self.para.db}/Results/{file_name}')
            with open(final_path, "a") as file_obj:
                file_obj.write(f"Network architecture: {self.para.run_config.network_architecture}\n"
                               f"Optimizer: {self.optimizer}\n"
                               f"Loss function: {self.criterion}\n"
                               f"Batch size: {self.para.batch_size}\n"
                               f"Balanced data: {self.para.balance_data}\n"
                               f"Number of epochs: {self.para.n_epochs}\n")
                # iterate over the layers of the neural net
                total_trainable_parameters = 0
                for layer in self.net.net_layers:
                    file_obj.write(f"\n")
                    try:
                        file_obj.write(f"Layer: {layer.name}\n")
                    except:
                        file_obj.write(f"Layer: {type(layer).__name__}\n")
                    file_obj.write(f"\n")
                    # get number of trainable parameters
                    layer_params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
                    total_trainable_parameters += layer_params
                    file_obj.write(f"Layer Trainable Parameters: {layer_params}\n")
                    try:
                        file_obj.write(f"Node labels: {layer.node_labels.num_unique_node_labels}\n")
                    except:
                        pass
                    try:
                        for i, n in enumerate(layer.n_properties):
                            file_obj.write(f"Number of Source Labels (type: {layer.source_label_descriptions[i]}) in channel {i}: {layer.n_source_labels[i]}\n")
                            file_obj.write(f"Number of Target Labels (type: {layer.target_label_descriptions[i]}) in channel {i}: {layer.n_target_labels[i]}\n")
                            if layer.bias_list[i]:
                                file_obj.write(f"Number of Bias Labels in channel {i}: {layer.n_bias_labels[i]}\n")
                            file_obj.write(f"Number of pairwise properties in channel {i}: {n}\n")
                            file_obj.write("\n")
                    except:
                        pass
                    try:
                        for i, n in enumerate(layer.n_node_labels):
                            file_obj.write(f"Number of Node Labels (type: {layer.node_label_descriptions[i]}) in channel {i}: {n}\n")
                            file_obj.write("\n")
                    except:
                        pass

                    weight_learnable_parameters = 0
                    bias_learnable_parameters = 0
                    try:
                        if layer.Param_W.requires_grad:
                            weight_learnable_parameters += layer.Param_W.numel()
                    except:
                        try:
                            if layer.lin.weight.requires_grad:
                                weight_learnable_parameters += layer.lin.weight.numel()
                        except:
                            pass
                    try:
                        if layer.Param_b.requires_grad:
                            bias_learnable_parameters += layer.Param_b.numel()
                    except:
                        try:
                            if layer.bias.requires_grad:
                                bias_learnable_parameters += layer.bias.numel()
                        except:
                            pass

                    file_obj.write("Weight matrix learnable parameters: {}\n".format(weight_learnable_parameters))
                    file_obj.write("Bias learnable parameters: {}\n".format(bias_learnable_parameters))
                    try:
                        file_obj.write(f"Edge labels: {layer.edge_labels.num_unique_edge_labels}\n")
                    except:
                        pass
                for name, param in self.net.named_parameters():
                    file_obj.write(f"Layer: {name} -> {param.requires_grad}\n")

                file_obj.write(f"\n")
                file_obj.write(f"Total trainable parameters: {total_trainable_parameters}\n")

        file_name = f'{self.para.db}_{self.para.config_id}_Results_run_id_{self.run_id}_validation_step_{self.para.validation_id}.csv'

        # if the file does not exist create a new file
        with open(self.results_path.joinpath(f'{self.para.db}/Results/{file_name}'), "w") as file_obj:
            file_obj.write("")

        # header use semicolon as delimiter
        if self.para.run_config.task == 'graph_regression':
            header = f"Dataset;Time;RunNumber;ValidationNumber;Seed;Epoch;TrainingSize;ValidationSize;TestSize;EpochLoss ({self.para.run_config.loss});" \
                     f"EpochAccuracy;EpochTime;EpochMAE;EpochMAEStd;ValidationLoss;ValidationAccuracy;ValidationMAE;ValidationMAEStd;TestLoss;TestAccuracy;TestMAE;TestMAEStd\n"
        else:
            if self.para.run_config.config.get('evaluation_metric', 'accuracy') == 'roc_auc':
                header = f"Dataset;Time;RunNumber;ValidationNumber;Seed;Epoch;TrainingSize;ValidationSize;TestSize;EpochLoss ({self.para.run_config.loss});" \
                         f"EpochAccuracy;EpochAUC;EpochTime;ValidationAccuracy;ValidationLoss;ValidationAUC;TestAccuracy;TestLoss;TestAUC\n"
            else:
                header = f"Dataset;Time;RunNumber;ValidationNumber;Seed;Epoch;TrainingSize;ValidationSize;TestSize;EpochLoss  ({self.para.run_config.loss});EpochAccuracy;" \
                         f"EpochTime;ValidationAccuracy;ValidationLoss;TestAccuracy;TestLoss\n"

        # Save file for results and add header if the file is new
        final_path = self.results_path.joinpath(f'{self.para.db}/Results/{file_name}')
        with open(final_path, "a") as file_obj:
            if os.stat(final_path).st_size == 0:
                file_obj.write(header)
        return True


    def postprocess_writer(self, epoch, epoch_time, train_values: EvaluationValues, validation_values: EvaluationValues, test_values: EvaluationValues):
        time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if self.para.print_results:
            # if class num is one print the mae and mse
            if self.para.run_config.task == 'graph_regression':
                print(
                    f'run: {self.run_id} val step: {self.k_val} epoch: {epoch + 1}/{self.para.n_epochs} epoch loss: {train_values.loss} epoch acc: {train_values.accuracy} epoch mae: {train_values.mae} +- {train_values.mae_std} epoch time: {epoch_time}'
                    f' validation acc: {validation_values.accuracy} validation loss: {validation_values.loss} validation mae: {validation_values.mae} +- {validation_values.mae_std}'
                    f'test acc: {test_values.accuracy} test loss: {test_values.loss} test mae: {test_values.mae} +- {test_values.mae_std}'
                    f'time: {epoch_time}')
            else:
                print(
                    f'run: {self.run_id} val step: {self.k_val} epoch: {epoch + 1}/{self.para.n_epochs} epoch loss: {train_values.loss} epoch acc: {train_values.accuracy}'
                    f' validation acc: {validation_values.accuracy} validation loss: {validation_values.loss}'
                    f'test acc: {test_values.accuracy} test loss: {test_values.loss}'
                    f'time: {epoch_time}')

        if self.para.run_config.task == 'graph_regression':
            res_str =   f"{self.para.db};{time};{self.run_id};{self.k_val};{self.seed};{epoch};{self.training_data.size};{self.validate_data.size};{self.test_data.size};" \
                        f"{train_values.loss};{train_values.accuracy};{epoch_time};{train_values.mae};{train_values.mae_std};" \
                        f"{validation_values.loss};{validation_values.accuracy};{validation_values.mae};{validation_values.mae_std};" \
                        f"{test_values.loss};{test_values.accuracy};{test_values.mae};{test_values.mae_std}\n"
        else:
            if self.para.run_config.config.get('evaluation_metric', 'accuracy') == 'roc_auc':
                res_str =   f"{self.para.db};{time};{self.run_id};{self.k_val};{self.seed};{epoch};{self.training_data.size};{self.validate_data.size};{self.test_data.size};" \
                            f"{train_values.loss};{train_values.accuracy};{train_values.accuracy_roc_auc};{epoch_time};" \
                            f"{validation_values.accuracy};{validation_values.loss};{validation_values.accuracy_roc_auc};" \
                            f"{test_values.accuracy};{test_values.loss};{test_values.accuracy_roc_auc}\n"
            else:
                res_str =   f"{self.para.db};{time};{self.run_id};{self.k_val};{self.seed};{epoch};{self.training_data.size};{self.validate_data.size};{self.test_data.size};" \
                            f"{train_values.loss};{train_values.accuracy};{epoch_time};" \
                            f"{validation_values.accuracy};{validation_values.loss};" \
                            f"{test_values.accuracy};{test_values.loss}\n"

        # Buffer CSV writes and flush periodically
        self._csv_buffer.append(res_str)
        if len(self._csv_buffer) >= self._csv_flush_interval or epoch == self.para.n_epochs - 1:
            self._flush_csv_buffer()

    def _flush_csv_buffer(self):
        if not self._csv_buffer:
            return
        file_name = f'{self.para.db}_{self.para.config_id}_Results_run_id_{self.run_id}_validation_step_{self.para.validation_id}.csv'
        final_path = self.results_path.joinpath(f'{self.para.db}/Results/{file_name}')
        with open(final_path, "a") as file_obj:
            file_obj.writelines(self._csv_buffer)
        self._csv_buffer.clear()


    def train_graph_task(self, epoch, values, train_batches, timer):
        loader = CustomBatchLoader(self.graph_data, train_batches)
        for batch_counter, batch in enumerate(loader, 0):
            batch_ids = train_batches[batch_counter]
            timer.measure("forward")
            self.optimizer.zero_grad()
            outputs = torch.zeros((len(batch), self.graph_data.num_classes), dtype=self.dtype).to(self.device)


            # Run ordinary GNN
            if not self.para.run_config.config.get('with_invariant_layers', True):
                timer.measure("forward_step")
                outputs = self.net(batch)
                timer.measure("forward_step")
            else:
                # Batch-wise processing for Share GNN is not possible (at the moment), so we process each graph individually
                for j, graph_id in enumerate(batch_ids, 0):
                    timer.measure("forward_step")
                    outputs[j] = self.net(self.graph_data[graph_id], pos=graph_id)
                    timer.measure("forward_step")

            # TODO run mixed models

            # calculate the loss
            if self.para.run_config.config.get('weighted_loss', False):
                self.set_loss_function(weight =self.class_weights[batch_counter])

            target_labels = self.graph_data.y[batch_ids]
            # check if output is two dimensional and task is graph classification
            if self.para.run_config.config.get('task', None) == 'graph_classification'  and len(outputs.shape) > 1 and outputs.shape[1] != 1:
                target_labels = torch.nn.functional.one_hot(self.graph_data.y[batch_ids], num_classes=self.graph_data.num_classes).to(self.dtype).to(self.device)
            elif self.para.run_config.config.get('task', None) == 'graph_regression' and outputs.shape[1] == 1:
                target_labels = target_labels.unsqueeze(1)
            loss = self.criterion(outputs, target_labels)
            timer.measure("forward")

            weights = []
            # save the weights to test if they are updated (only in debug mode)
            if self.para.save_weights:
                for i, layer in enumerate(self.net.net_layers):
                    weights.append([x.item() for x in layer.Param_W])

            timer.measure("backward")
            loss.backward()
            self.optimizer.step()
            timer.measure("backward")
            timer.reset()

            # test if the weights are updated (only in debug mode)
            if self.para.save_weights:
                self.test_weight_update(weights)

            epoch_values, validation_values, test_values = values
            epoch_values.loss += loss.item()

            # Get the training accuracy
            epoch_values, validation_values, test_values = self.evaluate_results(epoch=epoch, train_values=epoch_values,
                                                                                 validation_values=validation_values,
                                                                                 test_values=test_values,
                                                                                 evaluation_type='training',
                                                                                 outputs=outputs,
                                                                                 labels=self.graph_data.y[batch_ids],
                                                                                 batch_idx=batch_counter,
                                                                                 batch_length=len(batch_ids),
                                                                                 num_batches=len(train_batches),
                                                                                 batches=train_batches)

    def evaluate_graph_task(self, graph_ids):
        labels = self.graph_data.y[graph_ids]
        outputs = []

        with torch.no_grad():
            self.net.train(False)
            # split the graph ids into batches to avoid memory issues
            eval_batch_size = self.para.run_config.config.get('eval_batch_size', 512)
            batches = [graph_ids[i:i + eval_batch_size] for i in range(0, len(graph_ids), eval_batch_size)]
            # deduce output size from batches and graph data
            total_out_len = 0
            for batch in batches:
                total_out_len += len(batch)
                outputs = torch.zeros((total_out_len, self.graph_data.num_classes), dtype=self.dtype).to(self.device)
            loader = CustomBatchLoader(self.graph_data, batches)
            batch_counter = 0
            if not self.para.run_config.config.get('with_invariant_layers', True):
                # Run ordinary GNN
                for i, batch in enumerate(loader):
                    #print(f"Evaluating batch {i + 1}/{len(batches)}")
                    outputs[batch_counter:batch_counter + len(batch)] = self.net(batch_data=batch)
                    batch_counter += len(batch)
            else:
                # Run Share GNN
                for j, data_pos in enumerate(graph_ids):
                    outputs[j] = self.net(self.graph_data[data_pos], pos=data_pos)
        return labels, outputs

    def train_node_task(self, epoch, values, train_batches, random_variation_bool, timer):
        for batch_counter, batch in enumerate(train_batches, 0):
            timer.measure("forward")
            self.optimizer.zero_grad()
            timer.measure("forward_step")
            if random_variation_bool:
                mean = self.para.run_config.config['input_features']['random_variation'].get('mean', 0.0)
                std = self.para.run_config.config['input_features']['random_variation'].get('std', 0.1)
                if self.para.run_config.config.get('precision', 'double') == 'float':
                    random_variation = torch.normal(mean=mean, std=std, size=self.graph_data[0].x.size(),
                                                    dtype=torch.float)
                else:
                    random_variation = torch.normal(mean=mean, std=std, size=self.graph_data[0].x.size(),
                                                    dtype=torch.double)
                outputs = self.net(self.graph_data[0].x + random_variation, 0)
            else:
                outputs = self.net(self.graph_data[0].x, 0)
                timer.measure("forward_step")

            # calculate the loss
            # squeeze second dimension if it is one
            if outputs.shape[1] == 1:
                outputs = outputs.squeeze(1)
            loss = self.criterion(outputs[batch], self.graph_data.y[batch])
            timer.measure("forward")

            weights = []
            # save the weights to test if they are updated (only in debug mode)
            if self.para.save_weights:
                for i, layer in enumerate(self.net.net_layers):
                    weights.append([x.item() for x in layer.Param_W])

            timer.measure("backward")
            loss.backward()
            self.optimizer.step()
            timer.measure("backward")
            timer.reset()

            # test if the weights are updated (only in debug mode)
            if self.para.save_weights:
                self.test_weight_update(weights)

            epoch_values, validation_values, test_values = values
            epoch_values.loss += loss.item()

            # Get the training accuracy
            epoch_values, validation_values, test_values = self.evaluate_results(epoch=epoch, train_values=epoch_values,
                                                                                 validation_values=validation_values,
                                                                                 test_values=test_values,
                                                                                 evaluation_type='training',
                                                                                 outputs=outputs[batch],
                                                                                 labels=self.graph_data.y[batch],
                                                                                 batch_idx=batch_counter,
                                                                                 batch_length=len(batch),
                                                                                 num_batches=len(train_batches))

    def evaluate_node_task(self, data):
        labels = self.graph_data.y[data]

        # use torch no grad to save memory
        with torch.no_grad():
            self.net.train(False)
            outputs = self.net(self.graph_data[0].x, 0)
            # squeeze second dimension if it is one
            if outputs.shape[1] == 1:
                outputs = outputs.squeeze(1)
        return labels, outputs[data]


    def get_train_batches(self, seeds, epoch):
        """
        Get the training batches according to the sampling method
        :param seeds: Vector of seeds for shuffling the training data
        :param epoch: Current epoch
        :return: Return the training batches
        """
        # divide the whole training data into batches
        if self.para.run_config.config.get('training_data_sampling', None) is None or self.para.run_config.config[
            'training_data_sampling'].get('type', None) == 'default':
            shuffling_seed = seeds[epoch][self.k_val] * self.run_id + self.seed
            np.random.seed(shuffling_seed)
            np.random.shuffle(self.training_data)
            self.para.run_config.batch_size = min(self.para.run_config.batch_size, len(self.training_data))
            train_batches = np.array_split(self.training_data,
                                           self.training_data.size // self.para.run_config.batch_size)

        # sample the batches from the training data uniformly
        elif self.para.run_config.config['training_data_sampling'].get('type', None) == 'random':
            shuffling_seed = seeds[epoch][self.k_val] * self.run_id + self.seed
            np.random.seed(shuffling_seed)
            np.random.shuffle(self.training_data)
            self.para.run_config.batch_size = min(self.para.run_config.batch_size, len(self.training_data))
            # get random indices from the training data
            random_indices = np.random.choice(len(self.training_data), len(self.training_data), replace=True)
            train_batches = np.array_split(self.training_data[random_indices],
                                           self.training_data.size // self.para.run_config.batch_size)

        # sample the batches from the training data respecting the output class distribution
        elif self.para.run_config.config['training_data_sampling'].get('type', None) == 'balanced':
            balancing_factor = self.para.run_config.config['training_data_sampling'].get('factor', 0.5)
            total_samples_per_epoch = self.para.run_config.config['training_data_sampling'].get(
                'total_samples_per_epoch', 1)
            # get the class distribution of the training data
            unique_classes, class_indices, class_counts = torch.unique(self.graph_data.y[self.training_data],
                                                                       return_counts=True, return_inverse=True)
            indices_per_class = []
            for i in unique_classes:
                indices_per_class.append(np.where(class_indices == i)[0])
            random_indices_per_class = []
            balancing = [1 - balancing_factor, balancing_factor]
            for i in range(len(indices_per_class)):
                random_indices_per_class.append(np.random.choice(self.training_data[indices_per_class[i]],
                                                                 int(total_samples_per_epoch * self.training_data.size *
                                                                     balancing[i]), replace=True))
            # concatenate the random indices
            random_indices = np.concatenate(random_indices_per_class)
            shuffling_seed = seeds[epoch][self.k_val] * self.run_id + self.seed
            np.random.seed(shuffling_seed)
            np.random.shuffle(random_indices)
            train_batches = np.array_split(random_indices, self.training_data.size // self.para.run_config.batch_size)

        # undersampling the majority class
        elif self.para.run_config.config['training_data_sampling'].get('type', None) == 'undersampling':
            shuffling_seed = seeds[epoch][self.k_val] * self.run_id + self.seed
            np.random.seed(shuffling_seed)
            # get the class distribution of the training data
            unique_classes, class_indices, class_counts = torch.unique(self.graph_data.y[self.training_data],
                                                                       return_counts=True, return_inverse=True)
            minimum_class_count = torch.min(class_counts).item()
            indices_per_class = []
            for i in unique_classes:
                indices_per_class.append(np.where(class_indices == i)[0])
            random_indices_per_class = []
            for i in range(len(indices_per_class)):
                random_indices_per_class.append(
                    np.random.choice(indices_per_class[i], minimum_class_count, replace=False))
            # concatenate the random indices
            random_indices = np.concatenate(random_indices_per_class)
            np.random.shuffle(random_indices)
            train_batches = np.array_split(self.training_data[random_indices],
                                           random_indices.size // self.para.run_config.batch_size)



        # sort the graphs by the number of nodes
        elif self.para.run_config.config['training_data_sampling'].get('type', None) == 'curriculum':
            train_batches = curriculum_sampling(graph_data=self.graph_data,
                                                training_data=self.training_data,
                                                num_batches=self.para.run_config.config['training_data_sampling'].get(
                                                    'num_batches', (
                                                                len(self.training_data) - 1) // self.para.run_config.batch_size + 1),
                                                batch_size=self.para.run_config.batch_size,
                                                bucket_num=self.para.run_config.config['training_data_sampling'][
                                                    'bucket_num'],
                                                total_epochs=self.para.n_epochs,
                                                epoch=epoch,
                                                anti=self.para.run_config.config['training_data_sampling'].get('anti',
                                                                                                               False),
                                                exclusive=self.para.run_config.config['training_data_sampling'].get(
                                                    'exclusive', True))

        # sort the graphs by the number of edges
        elif self.para.run_config.config['training_data_sampling'].get('type', None) == 'curriculum_edges':
            train_batches = curriculum_sampling(graph_data=self.graph_data,
                                                training_data=self.training_data,
                                                num_batches=self.para.run_config.config['training_data_sampling'].get(
                                                    'num_batches', (
                                                                len(self.training_data) - 1) // self.para.run_config.batch_size + 1),
                                                batch_size=self.para.run_config.batch_size,
                                                bucket_num=self.para.run_config.config['training_data_sampling'][
                                                    'bucket_num'],
                                                total_epochs=self.para.n_epochs,
                                                epoch=epoch,
                                                anti=self.para.run_config.config['training_data_sampling'].get('anti',
                                                                                                               False),
                                                exclusive=self.para.run_config.config['training_data_sampling'].get(
                                                    'exclusive', True),
                                                use_edges=True)
        else:
            shuffling_seed = seeds[epoch][self.k_val] * self.run_id + self.seed
            np.random.seed(shuffling_seed)
            np.random.shuffle(self.training_data)
            self.para.run_config.batch_size = min(self.para.run_config.batch_size, len(self.training_data))
            train_batches = np.array_split(self.training_data,
                                           self.training_data.size // self.para.run_config.batch_size)
        return train_batches
