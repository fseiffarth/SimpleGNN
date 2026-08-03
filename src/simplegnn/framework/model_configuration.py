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
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple

import numpy as np
import pandas as pd
import sklearn
import torch
from torch import optim, nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR, LambdaLR

from simplegnn.datasets.graph_dataset import GraphDataset, GraphData, CustomBatchLoader
from simplegnn.framework.utils.data_sampling import curriculum_sampling
from simplegnn.framework.utils.parameters import Parameters
from simplegnn.models.model import GraphModel
from simplegnn.models.ShareGNN.layers.inv_based_message_passing import InvariantBasedMessagePassingLayer
from simplegnn.models.ShareGNN.layers.inv_based_pooling import InvariantBasedAggregationLayer
from simplegnn.models.ShareGNN.layers.inv_based_positional_encoding import InvariantBasedPositionalEncodingLayer
from simplegnn.utils.utils import get_k_lowest_nonzero_indices, valid_pruning_configuration, is_pruning
from simplegnn.utils.timer import TimeClass


# Aliases grouped exactly as in ModelConfiguration.set_loss_function, mapped to a
# short human-readable label for the per-epoch console line.
_LOSS_DISPLAY_NAMES = {
    'CrossEntropyLoss': 'CrossEntropy',
    'MeanSquaredError': 'MSE', 'MSELoss': 'MSE', 'mse': 'MSE', 'MSE': 'MSE',
    'RootedMeanSquaredError': 'RMSE', 'RMSELoss': 'RMSE', 'rmse': 'RMSE', 'RMSE': 'RMSE',
    'L1Loss': 'MAE', 'l1': 'MAE', 'L1': 'MAE', 'mean_absolute_error': 'MAE',
    'mae': 'MAE', 'MAE': 'MAE', 'MeanAbsoluteError': 'MAE',
    'SmoothL1Loss': 'SmoothL1', 'smooth_l1': 'SmoothL1', 'SmoothL1': 'SmoothL1',
    'Huber': 'SmoothL1', 'HuberLoss': 'SmoothL1', 'huber': 'SmoothL1',
    'BCELoss': 'BCE', 'bce': 'BCE', 'BCE': 'BCE',
    'BCEWithLogitsLoss': 'BCE', 'bce_with_logits': 'BCE', 'BCEWithLogits': 'BCE',
    'NLLLoss': 'NLL', 'nll': 'NLL', 'NLL': 'NLL',
}


def loss_display_name(loss) -> str:
    """
    Short display label for a configured loss string.

    Maps every loss alias accepted by :meth:`ModelConfiguration.set_loss_function`
    to a compact label (``MAE``, ``MSE``, ``RMSE``, ``SmoothL1``, ``CrossEntropy``,
    ``BCE``, ``NLL``). Unknown values are returned unchanged so the console line
    still names whatever was configured.

    Parameters
    ----------
    loss : str
        The configured loss string (``para.run_config.loss``).

    Returns
    -------
    str
        Display label for the loss.
    """
    return _LOSS_DISPLAY_NAMES.get(str(loss), str(loss))


def pooled_abs_error_stats(abs_err) -> Tuple[float, float]:
    """
    Mean and population standard deviation of a tensor of absolute errors.

    Computed over the pooled elements (not as an average of per-batch statistics),
    and returned as plain Python floats so they are written to the CSV as numbers
    rather than ``tensor(...)`` reprs.

    Parameters
    ----------
    abs_err : torch.Tensor
        Absolute errors ``|label - output|`` (any shape; flattened internally).

    Returns
    -------
    Tuple[float, float]
        ``(mae, std)``. ``std`` is ``0.0`` for a single element (population std,
        ``unbiased=False``), never ``NaN``. Empty input yields ``(0.0, 0.0)``.
    """
    abs_err = abs_err.detach().flatten()
    n = abs_err.numel()
    if n == 0:
        return 0.0, 0.0
    mae = abs_err.mean().item()
    if n == 1:
        return mae, 0.0
    return mae, abs_err.std(unbiased=False).item()


def binary_roc_auc(outputs, labels) -> float:
    """
    ROC-AUC of a binary classifier from its raw (unnormalized) network outputs.

    The score fed to :func:`sklearn.metrics.roc_auc_score` is the positive-class
    probability, not the hard ``argmax`` prediction. Ranking by hard 0/1 labels
    collapses the ROC curve to a single operating point and turns the AUC into
    balanced accuracy, which is not the metric OGB's ``rocauc`` leaderboards
    (ogbg-molhiv and friends) are scored on.

    Parameters
    ----------
    outputs : torch.Tensor
        Network outputs of shape ``(N, C)``. ``C == 2`` is the binary case and
        uses ``softmax(...)[:, 1]``; ``C == 1`` uses the raw logit (monotone in
        the sigmoid probability, so the AUC is identical).
    labels : torch.Tensor
        Ground-truth labels, either class indices ``(N,)`` or one-hot ``(N, C)``.

    Returns
    -------
    float
        The ROC-AUC, or the neutral ``0.5`` when it is undefined because the
        batch/split contains a single class. Single-class batches are common
        when training on a skewed dataset such as ogbg-molhiv, so this is
        checked up front rather than caught afterwards -- depending on the
        sklearn version ``roc_auc_score`` either raises or returns ``NaN`` with
        an ``UndefinedMetricWarning``, and a NaN would poison the running mean.
    """
    outputs = outputs.detach()
    labels = labels.detach()
    if labels.dim() > 1 and labels.shape[1] > 1:
        labels = torch.argmax(labels, dim=1)
    labels = labels.flatten()
    if labels.numel() == 0 or torch.unique(labels).numel() < 2:
        return 0.5
    if outputs.dim() > 1 and outputs.shape[1] > 1:
        scores = torch.softmax(outputs.float(), dim=1)[:, 1]
    else:
        scores = outputs.float().flatten()
    try:
        auc = float(sklearn.metrics.roc_auc_score(labels.cpu().numpy(), scores.cpu().numpy()))
    except ValueError:
        return 0.5
    return 0.5 if math.isnan(auc) else auc


def inverse_transform_targets(values, invert_cfg, stats):
    """
    Map normalized regression targets/outputs back to the original scale.

    Applies the inverse of the output normalization configured via
    ``invert_outputs``. Call once per tensor so that labels and outputs receive the
    *same* transform (a past bug applied ``minmax_zero`` to labels only, inflating
    the MAE).

    Parameters
    ----------
    values : torch.Tensor
        Normalized targets or model outputs.
    invert_cfg : dict or None
        The ``invert_outputs`` configuration. Only ``dict`` values with a
        ``normalization`` of ``standard`` / ``minmax`` / ``minmax_zero`` transform;
        anything else returns ``values`` unchanged.
    stats : dict
        Statistics of the original (un-normalized) targets with keys ``mean``,
        ``std``, ``min``, ``max``.

    Returns
    -------
    torch.Tensor
        Values on the original scale.
    """
    if not isinstance(invert_cfg, dict):
        return values
    normalization = invert_cfg.get('normalization', 'standard')
    if normalization == 'standard':
        return values * (stats['std'] + 1e-8) + stats['mean']
    if normalization == 'minmax':
        return values * (stats['max'] - stats['min'] + 1e-8) + stats['min']
    if normalization == 'minmax_zero':
        return (0.5 * values + 0.5) * (stats['max'] - stats['min'] + 1e-8) + stats['min']
    return values


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
        # Running accumulators for the pooled training MAE/std (summed over all
        # batches of the epoch, so mae/mae_std are the true pooled statistics
        # rather than an average of per-batch means/stds).
        self.sum_abs_err = 0.0
        self.sumsq_abs_err = 0.0
        self.n_abs_err = 0



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
        # cached full-graph forward output for node-level tasks (see evaluate_node_task)
        self._node_eval_outputs = None
        # cached statistics of the un-normalized targets (only used when
        # invert_outputs is configured); computed once via _get_original_y_stats
        self._original_y_stats = None
        # last computed validation/test metrics, carried forward to the CSV on
        # epochs where validation is skipped (validation_frequency > 1)
        self._last_validation_values = None
        self._last_test_values = None
        self._csv_buffer = []
        self._csv_flush_interval = self.para.run_config.config.get('csv_flush_interval', 10)
        # hash-keyed transfer (spec 18 B2/B3): cached sidecar payload (the
        # weight keys are weight-independent, so they are exported once per
        # configuration; False = checked and nothing to save) and the report
        # of an applied `transfer:` block
        self._transfer_keys_payload = None
        self.transfer_report = None
        # get gpu or cpu: (cpu is recommended at the moment)
        if self.para.run_config.config.get('device', None) is not None:
            self.device = torch.device(self.para.run_config.config['device'] if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        self.dtype = torch.float
        if self.para.run_config.config.get('precision', 'float') == 'double':
            self.dtype = torch.double
        # deterministic: True trades throughput for bit-exact reproducibility across
        # reruns of the same (seed, config) by pinning intra-op parallelism to 1
        # thread (torch.sparse.mm/index_add_ reduction order is otherwise thread-
        # schedule-dependent) and forcing deterministic kernels. Off by default -
        # the ShareGNN sparse-mm path is built for multi-core throughput (specs/07,
        # specs/08) and only needs this for debugging/verification runs.
        # torch_threads: explicit cap on PyTorch's intra-op thread pool for a
        # single serial run (run_configurations only forces OMP_NUM_THREADS=1
        # when parallelizing across MULTIPLE joblib workers; a lone job
        # otherwise defaults to all logical cores). On many-small-op ShareGNN
        # workloads (small graphs, small batches) that default is actively
        # harmful — measured on ZINC-full: 52 ms/batch at 8 threads vs.
        # 2,727 ms/batch at 24 threads on a 12-core/24-thread CPU, because
        # thread-pool synchronization overhead swamps the tiny per-op compute.
        # Off (no-op) unless set, so existing configs/behavior are unchanged.
        # Accepts 'auto' / -1 to adapt to the host instead of pinning a fixed
        # count: 'auto' uses half the logical cores (min 1). Full core count was
        # measured catastrophic on this many-small-op workload (52 ms/batch at 8
        # threads vs. 2,727 ms at 24 on a 24-thread CPU), so staying well below
        # core count matters more than squeezing out the last threads; half is a
        # simple, monotonic rule that grows with the machine (8 on 16-thread, 16
        # on 32-thread) without oversubscribing. An explicit integer is instead
        # clamped at the logical core count.
        torch_threads = self.para.run_config.config.get('torch_threads', None)
        if torch_threads is not None:
            cpu_count = os.cpu_count() or 1
            if isinstance(torch_threads, str) and torch_threads.lower() == 'auto' \
                    or torch_threads == -1:
                torch_threads = max(1, cpu_count // 2)
            else:
                torch_threads = min(int(torch_threads), cpu_count)
            torch.set_num_threads(torch_threads)
        if self.para.run_config.config.get('deterministic', False):
            torch.set_num_threads(1)
            torch.use_deterministic_algorithms(True)

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
        early_stopping.patience epochs, or if the optimizer's learning rate
        has decayed to or below early_stopping.lr_threshold (if configured).
        Both criteria require early_stopping.enabled: True. See
        early_stopping() for details.

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

        # Report the execution mode (device + batched or per-graph forward)
        print(f"Execution mode: {self._execution_mode_description()} (device: {self.device})")
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
            elif self.para.run_config.config['task'] in ['node_classification', 'node_regression']:
                self.train_node_task(epoch=epoch, values=(epoch_values, validation_values, test_values), train_batches=train_batches, timer=timer)
            else:
                raise ValueError(f"Task {self.para.run_config.config['task']} not implemented")

            # EpochLoss is the mean per-batch loss over the epoch. train_*_task
            # accumulates a running sum in epoch_values.loss, so divide by the
            # number of batches here to make it comparable to ValidationLoss.
            if len(train_batches) > 0:
                epoch_values.loss /= len(train_batches)

            # TODO Pruning
            if valid_pruning_configuration(self.para, epoch):
                self.model_pruning(epoch)

            # Evaluate the results on validation and test set. On epochs where
            # validation is skipped (validation_frequency > 1) carry the last
            # computed validation/test metrics forward instead of writing zeros.
            validation_frequency = self.para.run_config.config.get('validation_frequency', 1)
            is_validation_epoch = (epoch + 1) % validation_frequency == 0 or epoch == self.para.n_epochs - 1
            if is_validation_epoch and self.validate_data.size != 0:
                epoch_values, validation_values, test_values = self.evaluate_results(epoch=epoch,train_values=epoch_values, validation_values=validation_values, test_values=test_values, evaluation_type='validation')
                self._last_validation_values = validation_values
            elif self.validate_data.size != 0 and self._last_validation_values is not None:
                validation_values = self._last_validation_values
            # check wheter there is a test split
            if is_validation_epoch and self.test_data.size != 0:
                epoch_values, validation_values, test_values = self.evaluate_results(epoch=epoch,train_values=epoch_values, validation_values=validation_values, test_values=test_values, evaluation_type='test')
                self._last_test_values = test_values
            elif self.test_data.size != 0 and self._last_test_values is not None:
                test_values = self._last_test_values

            # Capture the LR actually used for this epoch's updates before the
            # scheduler steps -- stepping may change it for the *next* epoch.
            epoch_lr = self.optimizer.param_groups[0]['lr']

            # Step the scheduler AFTER validation, so ReduceLROnPlateau sees the
            # actual validation loss (previously it stepped on a stale 0.0).
            if self.scheduler is not None:
                if self.para.run_config.config['scheduler']['type'] == 'ReduceLROnPlateau':
                    self.scheduler.step(validation_values.loss)
                else:
                    self.scheduler.step()

            timer.measure("epoch")
            epoch_time = timer.get_flag_time("epoch")

            # Write the results to the results file
            self.postprocess_writer(epoch, epoch_time, epoch_values, validation_values, test_values, epoch_lr)


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
                    print(f"Accuracy: {accuracy} %")
            else:
                if do_print:
                    print(f"Evaluation completed for graph regression task.")
                    mae_error = torch.mean(torch.abs(target_values - target_outputs))
                    rsme_error = torch.mean(torch.sqrt((target_values - target_outputs) ** 2))
                    print(f"Mean Absolute Error: {mae_error}")
        elif self.para.run_config.task in ['node_classification', 'node_regression']:
            target_values, target_outputs = self.evaluate_node_task(graph_ids)
            if do_print and self.para.run_config.task == 'node_classification':
                predictions = torch.argmax(target_outputs, dim=1)
                accuracy = 100 * torch.sum(predictions == target_values).item() / len(target_values)
                print(f"Accuracy: {accuracy} %")
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
            # in-memory path (same-dataset only): use the given network verbatim
            print('Using pretrained network')
            self.net = pretrained_network
        else:
            self.net = GraphModel(graph_data=self.graph_data, para=self.para, seed=self.seed, device=self.device)
            # hash-keyed cross-dataset transfer (spec 18 B3): build the target
            # net normally, then remap a source checkpoint into it
            transfer_config = self.para.run_config.config.get('transfer', None)
            if transfer_config:
                self.transfer_report = self._apply_transfer_from_config(transfer_config)

        # set the network to device
        self.net.to(self.device)
        # move the graph data to the same device as the network (features,
        # labels, attributes); slices/num_nodes bookkeeping stays on CPU
        self.graph_data.to(self.device)
        # new weights: drop any cached node-task evaluation outputs
        self._node_eval_outputs = None
        print(f'Network initialized with seed {self.seed}')

    def _apply_transfer_from_config(self, transfer_config: dict):
        """
        Resolve the source checkpoint + ``.keys.pt`` sidecar named by a
        ``transfer:`` block, remap the checkpoint into the freshly built
        ``self.net`` via :func:`apply_transfer`, apply the freeze strategy,
        and persist the per-layer report (spec 18 B3).
        """
        from simplegnn.framework.utils.configuration_checks import (
            check_transfer_runtime_requirements)
        from simplegnn.framework.utils.transfer import (
            apply_transfer, apply_transfer_strategy, load_transfer_sidecar,
            sidecar_path_for)

        # spec 18 B4: schema re-validation plus the existence checks that can
        # only run now (source checkpoint, .keys.pt sidecar, target label
        # hash vocabularies). Only the label descriptions the invariant
        # layers consume are checked — graph_data.node_labels also holds the
        # dataset's in-memory 'primary' labels, which never carry hashes.
        required_labels = set()
        for layer in getattr(self.net, 'net_layers', None) or []:
            if not hasattr(layer, 'export_weight_keys'):
                continue
            for attr in ('source_label_descriptions', 'target_label_descriptions',
                         'node_label_descriptions'):
                required_labels.update(getattr(layer, attr, None) or [])
            if getattr(layer, 'bias', False):
                required_labels.update(getattr(layer, 'bias_label_descriptions', None) or [])
        checkpoint_path = check_transfer_runtime_requirements(
            transfer_config, self.graph_data, sorted(required_labels))
        print(f"Transfer: loading source checkpoint {checkpoint_path}")
        source_state_dict = torch.load(str(checkpoint_path), map_location='cpu', weights_only=True)
        match = (transfer_config.get('invariant_transfer') or {}).get('match', 'hashes')
        if match == 'none':
            # invariant layers are re-initialized; only name+shape copies of
            # the standard layers remain, which need no sidecar
            source_keys = {'schema': 1, 'layers': {}}
        else:
            source_keys = load_transfer_sidecar(sidecar_path_for(checkpoint_path))

        report = apply_transfer(self.net, source_state_dict, source_keys, transfer_config)
        report.source_checkpoint = str(checkpoint_path)
        apply_transfer_strategy(self.net, transfer_config, report)
        print(report.format())

        # append the report to the run's results
        report_dir = self.results_path.joinpath(f'{self.para.db}/TransferReports')
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir.joinpath(
            f'transfer_report_{self.para.config_id}_run_{self.run_id}_val_step_{self.k_val}.json')
        with open(report_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        return report

    def _save_transfer_sidecar(self, final_path: Path) -> None:
        """
        Save the portable ``<model>.keys.pt`` sidecar next to a best-model
        checkpoint (spec 18 B2).

        Guarded by ``save_transfer_keys`` in the hyperparameter config:
        ``False`` disables the sidecar, ``True`` forces it whenever invariant
        layers exist, and the default (unset) saves it when invariant layers
        with hash vocabularies are present. The keys are weight-independent,
        so they are exported once and re-saved per checkpoint.
        """
        flag = self.para.run_config.config.get('save_transfer_keys', None)
        if flag is False:
            return
        if self._transfer_keys_payload is None:
            export = getattr(self.net, 'export_transfer_keys', None)
            if export is None:
                self._transfer_keys_payload = False
                return
            try:
                payload = export()
            except Exception as e:
                print(f"⚠ Warning: could not export transfer keys ({e}); "
                      f"no .keys.pt sidecar will be saved")
                self._transfer_keys_payload = False
                return
            has_layers = bool(payload['layers'])
            has_hashes = any(head.get('has_hashes', False)
                             for keys in payload['layers'].values()
                             for head in keys.get('heads', []))
            if not has_layers or (flag is None and not has_hashes):
                # nothing to key, or default-off because no label file carries
                # hashes (v1 labels); an explicit True still saves what exists
                self._transfer_keys_payload = payload if (flag and has_layers) else False
            else:
                self._transfer_keys_payload = payload
        if self._transfer_keys_payload:
            from simplegnn.framework.utils.transfer import sidecar_path_for
            torch.save(self._transfer_keys_payload, str(sidecar_path_for(final_path)))



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
        elif self.para.run_config.loss in ['SmoothL1Loss', 'smooth_l1', 'SmoothL1', 'Huber', 'HuberLoss', 'huber']:
            # Huber/smooth-L1: quadratic near 0, linear beyond beta -- less
            # sensitive to outliers than MAE while still training toward it.
            # ValidationMAE/TestMAE are computed independently of the training
            # loss, so evaluation stays on the task metric regardless.
            self.criterion = nn.SmoothL1Loss(*args, **kwargs)
        elif self.para.run_config.loss in ['BCELoss', 'bce', 'BCE']:
            self.criterion = nn.BCELoss(*args, **kwargs)
        elif self.para.run_config.loss in ['BCEWithLogitsLoss', 'bce_with_logits', 'BCEWithLogits']:
            self.criterion = nn.BCEWithLogitsLoss(*args, **kwargs)
        elif self.para.run_config.loss in ['NLLLoss', 'nll', 'NLL']:
            self.criterion = nn.NLLLoss(*args, **kwargs)
        else:
            raise ValueError(f"Loss function {self.para.run_config.loss} not implemented")

    def apply_l1_proximal(self):
        """
        Proximal L1 (soft-thresholding) on invariant-layer ``Param_W``.

        Called immediately after ``optimizer.step()``. A plain L1 term added to
        the loss does not produce clean sparsity under Adam/AdamW (the constant
        subgradient gets rescaled per coordinate by the adaptive denominator), so
        instead we take the explicit proximal step

            ``w <- sign(w) * relu(|w| - lr * lambda)``

        which drives uninformative rule weights to *exactly* 0 (learned rule
        selection, the data-driven cousin of ``rule_occurrence_threshold``). The
        threshold is scaled by the current learning rate so it stays consistent
        as the scheduler decays ``lr``; the effective per-step magnitude shrink
        is therefore ``lr * lambda``.

        Notes
        -----
        Configuration (all keys optional; 0 / absent = off for that group):

        ```yaml
        l1_regularization: { convolution: 0.01, aggregation: 0.01, encoding: 0.01 }
        ```

        - ``convolution`` -> lambda for InvariantBasedMessagePassingLayer.Param_W
        - ``aggregation``  -> lambda for InvariantBasedAggregationLayer.Param_W
        - ``encoding``     -> lambda for InvariantBasedPositionalEncodingLayer.Param_W

        ``encoding`` prunes embedding-table entries rather than message-passing
        rules, so it is not "the data-driven cousin of ``rule_occurrence_threshold``"
        in the same sense as ``convolution``/``aggregation`` -- it is the network's
        input embedding, so an overly aggressive lambda can zero out capacity before
        it has learned anything useful. Tune it separately from the other two.

        Calibrating lambda: a weight receiving no counteracting gradient loses
        ``lr * lambda`` per step, i.e. ``steps_per_epoch * lr * lambda`` per epoch.
        To zero unused ~1e-3 weights over roughly one epoch with ~80 batches at
        ``lr=1e-3``: ``lambda ~ 1e-3 / (80 * 1e-3) ~ 0.01``. So a sensible sweep is
        around {0.002, 0.01, 0.05}; ``lambda >= 1`` is far too aggressive (zeros a
        0.003 weight in ~3 steps, before gradients can rescue useful rules).
        Uses ``torch.no_grad`` and an in-place update so autograd is untouched.
        """
        l1_cfg = self.para.run_config.config.get('l1_regularization', None)
        if not l1_cfg:
            return
        conv_lambda = float(l1_cfg.get('convolution', 0.0) or 0.0)
        aggr_lambda = float(l1_cfg.get('aggregation', 0.0) or 0.0)
        enc_lambda = float(l1_cfg.get('encoding', 0.0) or 0.0)
        if conv_lambda <= 0.0 and aggr_lambda <= 0.0 and enc_lambda <= 0.0:
            return

        lr = self.optimizer.param_groups[0]['lr']
        with torch.no_grad():
            for layer in self.net.net_layers:
                if isinstance(layer, InvariantBasedMessagePassingLayer):
                    lam = conv_lambda
                elif isinstance(layer, InvariantBasedAggregationLayer):
                    lam = aggr_lambda
                elif isinstance(layer, InvariantBasedPositionalEncodingLayer):
                    lam = enc_lambda
                else:
                    continue
                if lam <= 0.0:
                    continue
                thresh = lr * lam
                w = layer.Param_W
                # softshrink(w, thresh) == sign(w) * relu(|w| - thresh): same
                # proximal update as one fused op instead of 4 (~17x faster,
                # bit-identical — verified against the old expression)
                w.copy_(F.softshrink(w, thresh))

    def set_optimizer(self):
        """
        Configure optimizer from configuration.

        Sets self.optimizer to the specified optimizer (Adam, SGD, AdamW,
        etc.) with learning rate and weight decay from configuration.

        Notes
        -----
        Optimizer type and hyperparameters are read from
        para.run_config.optimizer, which may be either:

        - a string naming the optimizer, e.g. ``AdamW`` (weight decay then comes
          from the separate top-level ``weight_decay`` key), or
        - a dict ``{ type: AdamW, weight_decay: 0.01, ... }`` where every field
          other than ``type`` is passed straight to the optimizer constructor
          (e.g. ``weight_decay``, ``betas``, ``eps``, ``amsgrad``, ``momentum``).
          Args given in the dict win over the top-level ``weight_decay`` key.

        Supported optimizers: Adam, SGD, AdamW, RMSprop, Adadelta, Adagrad.
        """
        opt_map = {'Adam': optim.Adam, 'AdamW': optim.AdamW, 'SGD': optim.SGD,
                   'RMSprop': optim.RMSprop, 'Adadelta': optim.Adadelta,
                   'Adagrad': optim.Adagrad}

        opt_spec = self.para.run_config.optimizer
        if isinstance(opt_spec, dict):
            opt_name = opt_spec.get('type', 'Adam')
            extra_kwargs = {k: v for k, v in opt_spec.items() if k != 'type'}
        else:
            opt_name = opt_spec
            extra_kwargs = {}

        opt = opt_map.get(opt_name, optim.Adam)

        kwargs = {'lr': self.para.learning_rate}
        # top-level weight_decay is the fallback; an explicit value in the
        # optimizer dict takes precedence.
        if 'weight_decay' not in extra_kwargs:
            kwargs['weight_decay'] = self.para.run_config.weight_decay
        kwargs.update(extra_kwargs)

        # keep run_config in sync so logging/metadata report the value actually used
        self.para.run_config.weight_decay = kwargs.get('weight_decay', 0.0)

        # only trainable parameters go into the optimizer: a transfer block's
        # freeze/linear_probe strategy sets requires_grad=False on transferred
        # layers (see framework.utils.transfer.apply_transfer_strategy)
        trainable = [p for p in self.net.parameters() if p.requires_grad]
        if not trainable:
            raise ValueError(
                "All model parameters are frozen — check the transfer.freeze globs / "
                "transfer.strategy configuration (linear_probe needs a trainable head).")
        self.optimizer = opt(trainable, **kwargs)

    def set_scheduler(self):
        """
        Configure learning rate scheduler from configuration.

        Sets self.scheduler to the specified LR scheduler (StepLR,
        ReduceLROnPlateau, etc.) if configured, otherwise None.

        Notes
        -----
        Scheduler type and parameters are read from
        para.run_config.config['scheduler'] if present.

        Supported schedulers: StepLR, ReduceLROnPlateau, CosineAnnealingLR,
        CosineWarmup.

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
            elif scheduler_type == 'CosineAnnealingLR':
                # Single cosine decay from the initial lr to eta_min over T_max
                # epochs (steps once per epoch via the else-branch in run_model).
                # T_max defaults to the run's epoch count -> one full half-cosine
                # so lr (and hence the L1 threshold lr*lambda) is high early and
                # anneals to eta_min late: aggressive rule pruning first, gentle
                # fine-tuning of the survivors after.
                t_max = scheduler.get('T_max', self.para.n_epochs)
                self.scheduler = CosineAnnealingLR(self.optimizer, T_max=t_max, eta_min=scheduler.get('eta_min', 0))
            elif scheduler_type == 'CosineWarmup':
                # GRIT/GraphGPS 'cosine_with_warmup': linear warmup from ~0 to the
                # base lr over num_warmup_epochs, then a half-cosine decay to 0 over
                # the remaining (max_epoch - num_warmup_epochs) epochs. Exact port of
                # HuggingFace get_cosine_schedule_with_warmup (num_cycles=0.5), which
                # is what GRIT/GPS's 'cosine_with_warmup' scheduler wraps -- see
                # graphgps/optimizer/extra_optimizers.py. Steps once per epoch via
                # the else-branch below, matching GRIT's epoch-stepped schedule.
                num_warmup_epochs = scheduler.get('num_warmup_epochs', 10)
                max_epoch = scheduler.get('max_epoch', self.para.n_epochs)

                def lr_lambda(current_epoch, num_warmup_epochs=num_warmup_epochs, max_epoch=max_epoch):
                    if current_epoch < num_warmup_epochs:
                        return max(1e-6, float(current_epoch) / float(max(1, num_warmup_epochs)))
                    progress = float(current_epoch - num_warmup_epochs) / float(max(1, max_epoch - num_warmup_epochs))
                    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

                self.scheduler = LambdaLR(self.optimizer, lr_lambda)


    def early_stopping(self, epoch):
        """
        Check whether training should stop before ``para.n_epochs``.

        Both gated by ``early_stopping.enabled`` in
        ``para.run_config.config``; ``enabled: False`` (the default) disables
        both criteria regardless of what else is set:

        - **Patience-based**: stop if the best validation epoch is more than
          ``patience`` epochs in the past.
        - **LR-based** (``lr_threshold`` set): stop once the optimizer's
          current learning rate drops to or below ``lr_threshold``. This
          mirrors the Dwivedi et al. "Benchmarking GNNs" ZINC protocol, where
          training with ``ReduceLROnPlateau`` stops once the scheduler has
          annealed the LR down to its floor rather than running out the full
          epoch budget. Optional -- omit ``lr_threshold`` to keep only the
          patience-based criterion.
        """
        es_config = self.para.run_config.config.get('early_stopping', {'enabled': False})
        if not es_config.get('enabled', False):
            return False
        if epoch - self.best_epoch["epoch"] > es_config['patience']:
            if self.para.print_results:
                print(f"Early stopping at epoch {epoch}: no validation improvement in "
                      f"{es_config['patience']} epochs")
            return True
        lr_threshold = es_config.get('lr_threshold', None)
        if lr_threshold is not None:
            current_lr = self.optimizer.param_groups[0]['lr']
            if current_lr <= lr_threshold:
                if self.para.print_results:
                    print(f"Early stopping at epoch {epoch}: learning rate {current_lr} "
                          f"<= lr_threshold {lr_threshold}")
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


    def _get_original_y_stats(self) -> dict:
        """
        Statistics (mean/std/min/max) of the un-normalized targets, cached.

        Used to invert output normalization when ``invert_outputs`` is configured.
        Computed once from ``self.graph_data.original_y`` and reused for
        every batch and every validation/test evaluation, instead of reducing the
        full target tensor on each call.
        """
        if self._original_y_stats is None:
            original_y = self.graph_data.original_y
            self._original_y_stats = {
                'mean': original_y.mean(),
                'std': original_y.std(),
                'min': original_y.min(),
                'max': original_y.max(),
            }
        return self._original_y_stats

    def _apply_inverse_transforms(self, flatten_labels, flatten_outputs):
        """
        Undo output feature/normalization transforms on labels and outputs.

        Single implementation shared by the training, validation and test
        evaluation branches. Applies ``output_features_inverse`` (via
        :meth:`GraphData.transform_data`) and then ``invert_outputs`` (via
        :func:`inverse_transform_targets`) to *both* tensors, so the regression
        metrics are always computed on the original scale.
        """
        config = self.para.run_config.config
        output_features_inverse = config.get('output_features_inverse', None)
        if output_features_inverse is not None:
            flatten_labels = GraphData.transform_data(flatten_labels, output_features_inverse)
            flatten_outputs = GraphData.transform_data(flatten_outputs, output_features_inverse)
        invert_outputs = config.get('invert_outputs', None)
        if invert_outputs is not None:
            stats = self._get_original_y_stats()
            flatten_labels = inverse_transform_targets(flatten_labels, invert_outputs, stats)
            flatten_outputs = inverse_transform_targets(flatten_outputs, invert_outputs, stats)
        return flatten_labels, flatten_outputs

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
            if self.para.run_config.task in ('graph_regression', 'node_regression'):
                # flatten the labels and outputs, undo any output normalization
                flatten_labels = labels.detach().flatten()
                flatten_outputs = outputs.detach().flatten()
                flatten_labels, flatten_outputs = self._apply_inverse_transforms(flatten_labels, flatten_outputs)
                # Accumulate pooled absolute-error statistics over the epoch so
                # mae/mae_std are the true pooled mean/std (not an average of
                # per-batch means/stds), stored as floats.
                abs_err = torch.abs(flatten_labels - flatten_outputs)
                train_values.sum_abs_err += abs_err.sum().item()
                train_values.sumsq_abs_err += (abs_err ** 2).sum().item()
                train_values.n_abs_err += abs_err.numel()
                n = train_values.n_abs_err
                train_values.mae = train_values.sum_abs_err / n if n else 0.0
                if n > 1:
                    variance = max(0.0, train_values.sumsq_abs_err / n - train_values.mae ** 2)
                    train_values.mae_std = float(np.sqrt(variance))
                else:
                    train_values.mae_std = 0.0
            else:
                prediction = torch.argmax(outputs, dim=1)
                batch_acc = 100 * torch.sum(prediction == labels).item() / len(labels)
                # accuracy
                train_values.accuracy = (train_values.accuracy * train_values.current_elements + batch_acc * batch_length) / (train_values.current_elements + batch_length)
                # roc_auc
                # This used to be gated on `training_data_sampling: undersampling`,
                # because a batch holding a single class makes roc_auc_score raise.
                # binary_roc_auc absorbs that case (returning the neutral 0.5), so
                # the gate only served to report a misleading EpochAUC of 0.0 for
                # every other sampling mode.
                if self.para.run_config.config.get('evaluation_metric', 'accuracy') == 'roc_auc':
                    batch_roc_auc = binary_roc_auc(outputs, labels)
                    train_values.accuracy_roc_auc = (train_values.accuracy_roc_auc * train_values.current_elements + batch_roc_auc * batch_length) / (train_values.current_elements + batch_length)
            train_values.current_elements += batch_length
            if self.para.print_results:
                # train_values.loss is the running sum over batches so far; show it
                # as a running mean and label it with the configured loss.
                loss_label = loss_display_name(self.para.run_config.loss)
                running_loss = train_values.loss / (batch_idx + 1)
                if self.graph_data.num_classes == 1 or self.para.run_config.task in ('graph_regression', 'node_regression'):
                    print("\tepoch: {}/{}, batch: {}/{}, {} loss: {:.4f}, MAE: {:.4f} ± {:.4f}".format(
                        epoch + 1, self.para.n_epochs, batch_idx + 1, num_batches,
                        loss_label, running_loss, train_values.mae, train_values.mae_std))
                else:
                    print("\tepoch: {}/{}, batch: {}/{}, {} loss: {:.4f}, acc: {:.2f} %".format(
                        epoch + 1, self.para.n_epochs, batch_idx + 1, num_batches,
                        loss_label, running_loss, batch_acc))
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
                    elif self.para.run_config.config.get('task', None) == 'graph_regression' and len(outputs.shape) > 1 and outputs.shape[1] == 1 and labels.dim() == 1:
                        labels = labels.unsqueeze(1)
                elif self.para.run_config.task in ['node_classification', 'node_regression']:
                    labels, outputs = self.evaluate_node_task(self.validate_data)
                else:
                    raise ValueError(f"Task {self.para.run_config.task} not implemented")
                # get validation loss
                validation_loss = self.criterion(outputs, labels).item()
                validation_values.loss = validation_loss
                if self.para.run_config.task in ('graph_regression', 'node_regression'):
                    flatten_labels = labels.detach().flatten()
                    flatten_outputs = outputs.detach().flatten()
                    flatten_labels, flatten_outputs = self._apply_inverse_transforms(flatten_labels, flatten_outputs)
                    validation_values.mae, validation_values.mae_std = pooled_abs_error_stats(
                        torch.abs(flatten_labels - flatten_outputs))
                else:
                    prediction = torch.argmax(outputs, dim=1)
                    if len(labels.shape) > 1:
                        labels = torch.argmax(labels, dim=1)
                    validation_acc = 100 * torch.sum(prediction==labels).item() / len(labels)
                    validation_values.accuracy = validation_acc
                    if self.para.run_config.config.get('evaluation_metric', 'accuracy') == 'roc_auc':
                        # roc_auc, ranked by the positive-class probability
                        validation_values.accuracy_roc_auc = binary_roc_auc(outputs, labels)

                # update best epoch
                if self.para.run_config.task in ('graph_regression', 'node_regression'):
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
                            self._save_transfer_sidecar(final_path)


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
                            self._save_transfer_sidecar(final_path)

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
                    elif self.para.run_config.config.get('task', None) == 'graph_regression' and len(outputs.shape) > 1 and outputs.shape[1] == 1 and labels.dim() == 1:
                        labels = labels.unsqueeze(1)
                elif self.para.run_config.task in ['node_classification', 'node_regression']:
                    labels, outputs = self.evaluate_node_task(self.test_data)
                else:
                    raise ValueError(f"Task {self.para.run_config.task} not implemented")

                test_loss = self.criterion(outputs, labels).item()
                test_values.loss = test_loss
                if self.para.run_config.task in ('graph_regression', 'node_regression'):
                    flatten_labels = labels.detach().flatten()
                    flatten_outputs = outputs.detach().flatten()
                    flatten_labels, flatten_outputs = self._apply_inverse_transforms(flatten_labels, flatten_outputs)
                    test_values.mae, test_values.mae_std = pooled_abs_error_stats(
                        torch.abs(flatten_labels - flatten_outputs))
                else:
                    prediction = torch.argmax(outputs, dim=1)
                    if len(labels.shape) > 1:
                        labels = torch.argmax(labels, dim=1)
                    test_acc = 100 * torch.sum(prediction == labels).item() / len(labels)
                    test_values.accuracy = test_acc
                    if self.para.run_config.config.get('evaluation_metric', 'accuracy') == 'roc_auc':
                        # roc_auc, ranked by the positive-class probability
                        test_values.accuracy_roc_auc = binary_roc_auc(outputs, labels)

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

    def collect_network_info(self) -> dict:
        """
        Collect a structured description of the trained network.

        Gathers the run configuration, the training hyperparameters and, for every
        layer of ``self.net``, its dimensions, trainable parameter counts and the
        ShareGNN label/property channel information (where available).

        Returns
        -------
        dict
            Dictionary with the keys ``db``, ``config_id``, ``task``, ``device``,
            ``precision``, ``seed``, ``network_architecture``, the training
            hyperparameters, a ``main_config`` view (dataset identity and resolved
            paths from the main YAML), a ``parameter_config`` view (training
            settings from the hyperparameter YAML), a ``layers`` list, a
            ``named_parameters`` list and ``total_trainable_parameters``.

        Notes
        -----
        Layer attributes are read defensively: layers that do not expose ShareGNN
        specific attributes (node/edge labels, pairwise properties) simply contribute
        empty channel lists.
        """
        run_config = self.para.run_config
        info = {
            'db': self.para.db,
            'config_id': self.para.config_id,
            'task': run_config.task,
            'device': str(self.device),
            'precision': run_config.config.get('precision', 'float'),
            'seed': self.seed,
            'network_architecture': run_config.network_architecture,
            'optimizer': str(self.optimizer),
            'loss': str(self.criterion),
            'learning_rate': self.para.learning_rate,
            'weight_decay': run_config.weight_decay,
            'dropout': run_config.dropout,
            'batch_size': run_config.batch_size,
            'balance_data': self.para.balance_data,
            'n_epochs': self.para.n_epochs,
            'layers': [],
            'named_parameters': [],
            'total_trainable_parameters': 0,
        }

        # Extract the merged experiment configuration into a main-config and a
        # parameter-config view for the report. ``run_config.config`` is the full
        # dict merged from the main, model and hyperparameter YAML files; the
        # loaded ``splits`` payload is deliberately excluded (only its path, held
        # under ``paths``, is kept) so the report stays small.
        config = run_config.config
        config_paths = config.get('paths', {}) or {}
        info['main_config'] = {
            'name': config.get('name'),
            'source': config.get('source'),
            'task': config.get('task', run_config.task),
            'paths': {key: str(value) for key, value in config_paths.items()},
        }
        info['parameter_config'] = {
            key: config.get(key)
            for key in (
                'precision', 'device', 'mode', 'optimizer', 'loss',
                'learning_rate', 'batch_size', 'epochs', 'weight_decay', 'dropout',
                'rule_occurrence_threshold', 'weight_initialization', 'scheduler',
                'early_stopping', 'training_data_sampling', 'input_features',
                'best_model', 'l1_regularization',
            )
            if config.get(key) is not None
        }

        for layer in self.net.net_layers:
            layer_info = {
                'name': getattr(layer, 'name', None) or type(layer).__name__,
                'class': type(layer).__name__,
                'in_features': getattr(layer, 'in_features', None),
                'out_features': getattr(layer, 'out_features', None),
                'in_channels': getattr(layer, 'in_channels', None),
                'out_channels': getattr(layer, 'out_channels', None),
                'trainable_parameters': sum(p.numel() for p in layer.parameters() if p.requires_grad),
                'node_labels': None,
                'edge_labels': None,
                'property_channels': [],
                'node_label_channels': [],
                'weight_parameters': 0,
                'bias_parameters': 0,
            }
            info['total_trainable_parameters'] += layer_info['trainable_parameters']

            try:
                layer_info['node_labels'] = layer.node_labels.num_unique_node_labels
            except AttributeError:
                pass
            try:
                layer_info['edge_labels'] = layer.edge_labels.num_unique_edge_labels
            except AttributeError:
                pass

            try:
                # Per-head-config weight parameter totals: weight_offset_description
                # carries one entry per (replica × property value), each tagged with
                # its originating head config, so summing 'weights' per 'head:' gives
                # that channel's weight parameter count (replicas included).
                weight_per_head = {}
                for d in getattr(layer, 'weight_offset_description', [])[1:]:
                    if isinstance(d, dict):
                        weight_per_head[d['head:']] = weight_per_head.get(d['head:'], 0) + d['weights']
                for i, n in enumerate(layer.n_properties):
                    has_bias = layer.bias_list[i]
                    # Bias params for the channel: in_features × unique bias labels,
                    # once per replicated head (matches _build_distributions).
                    bias_params = (layer.in_features * layer.n_bias_labels[i]
                                   * layer.n_heads_per_label[i]) if has_bias else 0
                    weight_params = weight_per_head.get(i, 0)
                    layer_info['property_channels'].append({
                        'source_label_type': layer.source_label_descriptions[i],
                        'n_source_labels': layer.n_source_labels[i],
                        'target_label_type': layer.target_label_descriptions[i],
                        'n_target_labels': layer.n_target_labels[i],
                        'n_bias_labels': layer.n_bias_labels[i] if has_bias else None,
                        'n_properties': n,
                        'weight_parameters': weight_params,
                        'bias_parameters': bias_params,
                        'trainable_parameters': weight_params + bias_params,
                    })
            except (AttributeError, IndexError):
                pass
            try:
                for i, n in enumerate(layer.n_node_labels):
                    layer_info['node_label_channels'].append({
                        'node_label_type': layer.node_label_descriptions[i],
                        'n_node_labels': n,
                    })
            except (AttributeError, IndexError):
                pass

            try:
                if layer.Param_W.requires_grad:
                    layer_info['weight_parameters'] += layer.Param_W.numel()
            except AttributeError:
                try:
                    if layer.lin.weight.requires_grad:
                        layer_info['weight_parameters'] += layer.lin.weight.numel()
                except AttributeError:
                    pass
            try:
                if layer.Param_b.requires_grad:
                    layer_info['bias_parameters'] += layer.Param_b.numel()
            except AttributeError:
                try:
                    if layer.bias.requires_grad:
                        layer_info['bias_parameters'] += layer.bias.numel()
                except AttributeError:
                    pass

            info['layers'].append(layer_info)

        for name, param in self.net.named_parameters():
            info['named_parameters'].append({
                'name': name,
                'shape': tuple(param.shape),
                'elements': param.numel(),
                'trainable': param.requires_grad,
            })
        return info

    @staticmethod
    def write_network_markdown(info: dict, final_path: Path):
        """
        Write the Markdown network report of the current model (overwrite mode).

        In contrast to the plain-text summary, the file is rewritten on every run, so
        it always describes exactly the model that is currently being trained.

        Parameters
        ----------
        info : dict
            Network description as returned by :meth:`collect_network_info`.
        final_path : Path
            Target ``*_Network.md`` file.
        """
        total = info['total_trainable_parameters']
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        def num(value):
            return f"{value:,}".replace(",", " ") if isinstance(value, int) else str(value)

        def dim(layer):
            features = f"{layer['in_features']} → {layer['out_features']}"
            if layer['in_channels'] == 1 and layer['out_channels'] == 1:
                return features
            return f"{features} ({layer['in_channels']} → {layer['out_channels']} ch)"

        def tensor_shape(channels, features):
            # Matches the framework's (C, N, F) / (N, F) convention (see
            # FrameworkLayer docstring); N (node count) is batch-dependent
            # and shown symbolically.
            if channels == 1:
                return f"(N, {features})"
            return f"({channels}, N, {features})"

        def tensor_dim(layer):
            return (f"{tensor_shape(layer['in_channels'], layer['in_features'])} → "
                    f"{tensor_shape(layer['out_channels'], layer['out_features'])}")

        lines = [
            f"# Network report: {info['db']} · {info['config_id']}",
            "",
            f"*Generated {timestamp} · task `{info['task']}` · device `{info['device']}` · "
            f"`{info['precision']}` precision · seed `{info['seed']}`*",
            "",
            "## Summary",
            "",
            "| | |",
            "|---|---|",
            f"| Dataset | `{info['db']}` |",
            f"| Configuration | `{info['config_id']}` |",
            f"| Layers | {len(info['layers'])} |",
            f"| Total trainable parameters | **{num(total)}** |",
            "",
            "## Training setup",
            "",
            "| Parameter | Value |",
            "|---|---|",
            f"| Learning rate | {info['learning_rate']} |",
            f"| Weight decay | {info['weight_decay']} |",
            f"| Dropout | {info['dropout']} |",
            f"| Batch size | {info['batch_size']} |",
            f"| Epochs | {info['n_epochs']} |",
            f"| Balanced data | {info['balance_data']} |",
            "",
            "<details><summary>Optimizer and loss function</summary>",
            "",
            "```",
            info['optimizer'],
            "",
            f"Loss function: {info['loss']}",
            "```",
            "",
            "</details>",
            "",
        ]

        def fmt_cfg(value):
            if isinstance(value, (dict, list)):
                # inline JSON, pipes escaped so the value stays inside its table cell
                return f"`{json.dumps(value, default=str).replace('|', chr(92) + '|')}`"
            return f"`{value}`"

        main_cfg = info.get('main_config') or {}
        if main_cfg:
            lines += [
                "## Main config",
                "",
                "*Dataset and paths resolved from the main YAML.*",
                "",
                "| | |",
                "|---|---|",
                f"| Dataset | `{main_cfg.get('name')}` |",
                f"| Source | `{main_cfg.get('source')}` |",
                f"| Task | `{main_cfg.get('task')}` |",
                "",
            ]
            cfg_paths = main_cfg.get('paths') or {}
            if cfg_paths:
                lines += [
                    "| Path | Location |",
                    "|---|---|",
                ]
                for key, value in cfg_paths.items():
                    lines.append(f"| {key} | `{value}` |")
                lines.append("")

        param_cfg = info.get('parameter_config') or {}
        if param_cfg:
            lines += [
                "## Parameter config",
                "",
                "*Training settings from the hyperparameter YAML. List-valued entries "
                "are the grid-search space; the values actually used for this run are in "
                "**Training setup** above.*",
                "",
                "| Parameter | Value |",
                "|---|---|",
            ]
            for key, value in param_cfg.items():
                label = key.replace('_', ' ').capitalize()
                lines.append(f"| {label} | {fmt_cfg(value)} |")
            lines.append("")

        lines += [
            "## Architecture",
            "",
            "| # | Layer | Class | Dimensions | Tensor dimensions | Trainable parameters | Share |",
            "|---:|---|---|---|---|---:|---:|",
        ]
        for i, layer in enumerate(info['layers']):
            share = 100 * layer['trainable_parameters'] / total if total else 0.0
            lines.append(f"| {i} | {layer['name']} | `{layer['class']}` | {dim(layer)} | "
                         f"{tensor_dim(layer)} | {num(layer['trainable_parameters'])} | {share:.1f} % |")
        lines += [
            f"| | **Total** | | | | **{num(total)}** | 100.0 % |",
            "",
            "```mermaid",
            "flowchart LR",
        ]
        for i, layer in enumerate(info['layers']):
            label = str(layer['name']).replace('"', "'")
            lines.append(f'    L{i}["{i}: {label}<br/>{dim(layer)}"]')
        if info['layers']:
            lines.append("    " + " --> ".join(f"L{i}" for i in range(len(info['layers']))))
        lines += [
            "```",
            "",
            "## Layer details",
            "",
        ]
        for i, layer in enumerate(info['layers']):
            lines += [
                f"### {i} · {layer['name']}",
                "",
                "| | |",
                "|---|---|",
                f"| Class | `{layer['class']}` |",
                f"| Dimensions | {dim(layer)} |",
                f"| Tensor dimensions | {tensor_dim(layer)} |",
                f"| Trainable parameters | {num(layer['trainable_parameters'])} |",
                f"| Weight matrix parameters | {num(layer['weight_parameters'])} |",
                f"| Bias parameters | {num(layer['bias_parameters'])} |",
            ]
            if layer['node_labels'] is not None:
                lines.append(f"| Unique node labels | {num(layer['node_labels'])} |")
            if layer['edge_labels'] is not None:
                lines.append(f"| Unique edge labels | {num(layer['edge_labels'])} |")
            lines.append("")
            if layer['property_channels']:
                lines += [
                    "| Channel | Source labels | Target labels | Bias labels | Pairwise properties | Trainable parameters |",
                    "|---:|---|---|---:|---:|---:|",
                ]
                for c, channel in enumerate(layer['property_channels']):
                    bias = num(channel['n_bias_labels']) if channel['n_bias_labels'] is not None else "–"
                    trainable = channel.get('trainable_parameters')
                    trainable_cell = num(trainable) if trainable is not None else "–"
                    lines.append(f"| {c} | {num(channel['n_source_labels'])} "
                                 f"(`{channel['source_label_type']}`) | "
                                 f"{num(channel['n_target_labels'])} (`{channel['target_label_type']}`) | "
                                 f"{bias} | {num(channel['n_properties'])} | {trainable_cell} |")
                lines.append("")
            if layer['node_label_channels']:
                lines += [
                    "| Channel | Node labels |",
                    "|---:|---|",
                ]
                for c, channel in enumerate(layer['node_label_channels']):
                    lines.append(f"| {c} | {num(channel['n_node_labels'])} "
                                 f"(`{channel['node_label_type']}`) |")
                lines.append("")

        lines += [
            "## Parameter tensors",
            "",
            "<details><summary>All named parameters</summary>",
            "",
            "| Parameter | Shape | Elements | Trainable |",
            "|---|---|---:|:---:|",
        ]
        for param in info['named_parameters']:
            shape = " × ".join(str(s) for s in param['shape']) or "scalar"
            lines.append(f"| `{param['name']}` | {shape} | {num(param['elements'])} | "
                         f"{'yes' if param['trainable'] else 'no'} |")
        lines += [
            "",
            "</details>",
            "",
            "## Network architecture (config)",
            "",
            "```json",
            json.dumps(info['network_architecture'], indent=2, default=str),
            "```",
            "",
        ]

        with open(final_path, "w") as file_obj:
            file_obj.write("\n".join(lines))

    def preprocess_writer(self)-> bool:
        if self.run_id == 0 and self.k_val == 0:
            # collect the net details (architecture, optimizer, learning rate, loss function, batch size,
            # number of epochs, balanced data, dropout) and write them as markdown
            network_info = self.collect_network_info()
            results_dir = self.results_path.joinpath(f'{self.para.db}/Results')
            self.write_network_markdown(network_info,
                                        results_dir.joinpath(f'{self.para.db}_{self.para.config_id}_Network.md'))

        file_name = f'{self.para.db}_{self.para.config_id}_Results_run_id_{self.run_id}_validation_step_{self.para.validation_id}.csv'

        # if the file does not exist create a new file
        with open(self.results_path.joinpath(f'{self.para.db}/Results/{file_name}'), "w") as file_obj:
            file_obj.write("")

        # header use semicolon as delimiter
        if self.para.run_config.task in ('graph_regression', 'node_regression'):
            header = f"Dataset;Time;RunNumber;ValidationNumber;Seed;Epoch;TrainingSize;ValidationSize;TestSize;EpochLoss ({self.para.run_config.loss});" \
                     f"EpochAccuracy;EpochTime;LearningRate;EpochMAE;EpochMAEStd;ValidationLoss;ValidationAccuracy;ValidationMAE;ValidationMAEStd;TestLoss;TestAccuracy;TestMAE;TestMAEStd\n"
        else:
            if self.para.run_config.config.get('evaluation_metric', 'accuracy') == 'roc_auc':
                header = f"Dataset;Time;RunNumber;ValidationNumber;Seed;Epoch;TrainingSize;ValidationSize;TestSize;EpochLoss ({self.para.run_config.loss});" \
                         f"EpochAccuracy;EpochAUC;EpochTime;LearningRate;ValidationAccuracy;ValidationLoss;ValidationAUC;TestAccuracy;TestLoss;TestAUC\n"
            else:
                header = f"Dataset;Time;RunNumber;ValidationNumber;Seed;Epoch;TrainingSize;ValidationSize;TestSize;EpochLoss  ({self.para.run_config.loss});EpochAccuracy;" \
                         f"EpochTime;LearningRate;ValidationAccuracy;ValidationLoss;TestAccuracy;TestLoss\n"

        # Save file for results and add header if the file is new
        final_path = self.results_path.joinpath(f'{self.para.db}/Results/{file_name}')
        with open(final_path, "a") as file_obj:
            if os.stat(final_path).st_size == 0:
                file_obj.write(header)
        return True


    def postprocess_writer(self, epoch, epoch_time, train_values: EvaluationValues, validation_values: EvaluationValues, test_values: EvaluationValues, epoch_lr):
        time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if self.para.print_results:
            # Console line reflects the configured loss (label) and the task's
            # reported metric: MAE for regression, accuracy/AUC for classification.
            loss_label = loss_display_name(self.para.run_config.loss)
            prefix = f'run: {self.run_id} val step: {self.k_val} epoch: {epoch + 1}/{self.para.n_epochs}'
            if self.para.run_config.task in ('graph_regression', 'node_regression'):
                print(
                    f'{prefix} | {loss_label} loss: {train_values.loss:.4f} | '
                    f'train MAE: {train_values.mae:.4f} ± {train_values.mae_std:.4f} | '
                    f'val MAE: {validation_values.mae:.4f} (loss {validation_values.loss:.4f}) | '
                    f'test MAE: {test_values.mae:.4f} (loss {test_values.loss:.4f}) | '
                    f'time: {epoch_time:.2f}s')
            else:
                metric = 'AUC' if self.para.run_config.config.get('evaluation_metric', 'accuracy') == 'roc_auc' else 'acc'
                if metric == 'AUC':
                    train_metric, val_metric, test_metric = (train_values.accuracy_roc_auc,
                                                             validation_values.accuracy_roc_auc,
                                                             test_values.accuracy_roc_auc)
                    metric_fmt = '.4f'
                else:
                    train_metric, val_metric, test_metric = (train_values.accuracy,
                                                             validation_values.accuracy,
                                                             test_values.accuracy)
                    metric_fmt = '.2f'
                print(
                    f'{prefix} | {loss_label} loss: {train_values.loss:.4f} | '
                    f'train {metric}: {train_metric:{metric_fmt}} | '
                    f'val {metric}: {val_metric:{metric_fmt}} (loss {validation_values.loss:.4f}) | '
                    f'test {metric}: {test_metric:{metric_fmt}} (loss {test_values.loss:.4f}) | '
                    f'time: {epoch_time:.2f}s')

        if self.para.run_config.task in ('graph_regression', 'node_regression'):
            res_str =   f"{self.para.db};{time};{self.run_id};{self.k_val};{self.seed};{epoch};{self.training_data.size};{self.validate_data.size};{self.test_data.size};" \
                        f"{train_values.loss};{train_values.accuracy};{epoch_time};{epoch_lr};{train_values.mae};{train_values.mae_std};" \
                        f"{validation_values.loss};{validation_values.accuracy};{validation_values.mae};{validation_values.mae_std};" \
                        f"{test_values.loss};{test_values.accuracy};{test_values.mae};{test_values.mae_std}\n"
        else:
            if self.para.run_config.config.get('evaluation_metric', 'accuracy') == 'roc_auc':
                res_str =   f"{self.para.db};{time};{self.run_id};{self.k_val};{self.seed};{epoch};{self.training_data.size};{self.validate_data.size};{self.test_data.size};" \
                            f"{train_values.loss};{train_values.accuracy};{train_values.accuracy_roc_auc};{epoch_time};{epoch_lr};" \
                            f"{validation_values.accuracy};{validation_values.loss};{validation_values.accuracy_roc_auc};" \
                            f"{test_values.accuracy};{test_values.loss};{test_values.accuracy_roc_auc}\n"
            else:
                res_str =   f"{self.para.db};{time};{self.run_id};{self.k_val};{self.seed};{epoch};{self.training_data.size};{self.validate_data.size};{self.test_data.size};" \
                            f"{train_values.loss};{train_values.accuracy};{epoch_time};{epoch_lr};" \
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


    def _share_gnn_batched_enabled(self) -> bool:
        """True if the batched ShareGNN forward is enabled. Batched is the
        default; disable via ``share_gnn_forward: {batched: false}`` in the
        configuration."""
        forward_config = self.para.run_config.config.get('share_gnn_forward', None) or {}
        return bool(forward_config.get('batched', True))

    def _execution_mode_description(self) -> str:
        """Human-readable execution mode: 'cpu', 'batched cpu', 'gpu' or
        'batched gpu'."""
        device_name = 'gpu' if self.device.type == 'cuda' else 'cpu'
        if not self.para.run_config.config.get('with_invariant_layers', True):
            # classical GNNs always process whole batches jointly
            return f'batched {device_name}'
        return f'batched {device_name}' if self._share_gnn_batched_enabled() else device_name

    def _assemble_share_gnn_batch(self, graph_ids) -> Tuple[SimpleNamespace, list]:
        """
        Concatenate the node features of all graphs in the batch (in order,
        duplicate ids allowed) and move them to the execution device in a
        single transfer, so the whole batch is loaded to the GPU together.
        Returns (batch_data, positions) for the batched ShareGNN forward.
        """
        slices = self.graph_data.slices['x']
        x = self.graph_data.x
        positions = [int(g) for g in graph_ids]
        x_cat = torch.cat([x[int(slices[g]):int(slices[g + 1])] for g in positions])
        if x_cat.device != self.device:
            x_cat = x_cat.to(self.device)
        return SimpleNamespace(x=x_cat), positions

    def train_graph_task(self, epoch, values, train_batches, timer):
        with_invariant_layers = self.para.run_config.config.get('with_invariant_layers', True)
        batched_share_gnn = with_invariant_layers and self._share_gnn_batched_enabled()
        if batched_share_gnn:
            # the batch features are assembled directly from the collated
            # dataset tensors, no PyG batch collation needed
            loader = train_batches
        else:
            loader = CustomBatchLoader(self.graph_data, train_batches)
        for batch_counter, batch in enumerate(loader, 0):
            batch_ids = train_batches[batch_counter]
            timer.measure("forward")
            self.optimizer.zero_grad(set_to_none=True)
            outputs = torch.zeros((len(batch), self.graph_data.num_classes), dtype=self.dtype).to(self.device)


            # Run ordinary GNN
            if not with_invariant_layers:
                timer.measure("forward_step")
                outputs = self.net(batch)
                timer.measure("forward_step")
            elif batched_share_gnn:
                # Batched Share GNN: all graphs of the batch are processed jointly
                timer.measure("forward_step")
                batch_data, positions = self._assemble_share_gnn_batch(batch_ids)
                outputs = self.net(batch_data, pos=positions)
                timer.measure("forward_step")
            else:
                # Unbatched Share GNN: process each graph individually
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
            elif self.para.run_config.config.get('task', None) == 'graph_regression' and outputs.shape[1] == 1 and target_labels.dim() == 1:
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
            self.apply_l1_proximal()
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
            # allocate the output tensor once (size is known up front)
            outputs = torch.zeros((len(graph_ids), self.graph_data.num_classes), dtype=self.dtype, device=self.device)
            if not self.para.run_config.config.get('with_invariant_layers', True):
                # Run ordinary GNN; split the graph ids into batches to avoid memory issues
                eval_batch_size = self.para.run_config.config.get('eval_batch_size', 512)
                batches = [graph_ids[i:i + eval_batch_size] for i in range(0, len(graph_ids), eval_batch_size)]
                loader = CustomBatchLoader(self.graph_data, batches)
                batch_counter = 0
                for i, batch in enumerate(loader):
                    outputs[batch_counter:batch_counter + len(batch)] = self.net(batch_data=batch)
                    batch_counter += len(batch)
            elif self._share_gnn_batched_enabled():
                # Batched Share GNN: evaluate in eval_batch_size chunks
                eval_batch_size = self.para.run_config.config.get('eval_batch_size', 512)
                batch_counter = 0
                for i in range(0, len(graph_ids), eval_batch_size):
                    batch_data, positions = self._assemble_share_gnn_batch(graph_ids[i:i + eval_batch_size])
                    outputs[batch_counter:batch_counter + len(positions)] = self.net(batch_data, pos=positions)
                    batch_counter += len(positions)
            else:
                # Run Share GNN (per-graph forward; the batch loader is not used here)
                for j, data_pos in enumerate(graph_ids):
                    outputs[j] = self.net(self.graph_data[data_pos], pos=data_pos)
        return labels, outputs

    def train_node_task(self, epoch, values, train_batches, timer):
        """
        One training epoch for node-level tasks (node_classification /
        node_regression).

        Node tasks operate on a single graph (graph 0 of the dataset): the
        train/validation/test splits contain *node* indices instead of graph
        indices. Every batch needs a full-graph forward pass (the model output
        for all nodes), from which the batch rows are selected for the loss.
        Random input variation is applied inside GraphModel.forward (only in
        training mode), so no noise handling is needed here.
        """
        graph = self.graph_data[0]
        for batch_counter, batch in enumerate(train_batches, 0):
            timer.measure("forward")
            self.optimizer.zero_grad(set_to_none=True)
            timer.measure("forward_step")
            outputs = self.net(graph, pos=0)
            timer.measure("forward_step")

            # calculate the loss
            # squeeze second dimension if it is one (single-target regression)
            if outputs.dim() > 1 and outputs.shape[1] == 1:
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
            self.apply_l1_proximal()
            # the weights changed: cached full-graph evaluation outputs are stale
            self._node_eval_outputs = None
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
        """
        Evaluate the model on a set of node indices.

        The full-graph forward output is cached in ``self._node_eval_outputs``
        and invalidated whenever the weights change (optimizer step, model
        initialization), so consecutive validation and test evaluations of the
        same epoch share one forward pass.
        """
        labels = self.graph_data.y[data]

        if self._node_eval_outputs is None:
            # use torch no grad to save memory
            with torch.no_grad():
                self.net.train(False)
                outputs = self.net(self.graph_data[0], pos=0)
                # squeeze second dimension if it is one (single-target regression)
                if outputs.dim() > 1 and outputs.shape[1] == 1:
                    outputs = outputs.squeeze(1)
            self._node_eval_outputs = outputs
        return labels, self._node_eval_outputs[data]


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
            shuffling_seed = seeds[epoch][self.k_val] + self.run_id * seeds.size + self.seed
            np.random.seed(shuffling_seed)
            np.random.shuffle(self.training_data)
            self.para.run_config.batch_size = min(self.para.run_config.batch_size, len(self.training_data))
            train_batches = np.array_split(self.training_data,
                                           self.training_data.size // self.para.run_config.batch_size)

        # sample the batches from the training data uniformly
        elif self.para.run_config.config['training_data_sampling'].get('type', None) == 'random':
            shuffling_seed = seeds[epoch][self.k_val] + self.run_id * seeds.size + self.seed
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
            shuffling_seed = seeds[epoch][self.k_val] + self.run_id * seeds.size + self.seed
            np.random.seed(shuffling_seed)
            np.random.shuffle(random_indices)
            train_batches = np.array_split(random_indices, self.training_data.size // self.para.run_config.batch_size)

        # undersampling the majority class
        elif self.para.run_config.config['training_data_sampling'].get('type', None) == 'undersampling':
            shuffling_seed = seeds[epoch][self.k_val] + self.run_id * seeds.size + self.seed
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
            shuffling_seed = seeds[epoch][self.k_val] + self.run_id * seeds.size + self.seed
            np.random.seed(shuffling_seed)
            np.random.shuffle(self.training_data)
            self.para.run_config.batch_size = min(self.para.run_config.batch_size, len(self.training_data))
            train_batches = np.array_split(self.training_data,
                                           self.training_data.size // self.para.run_config.batch_size)
        return train_batches
