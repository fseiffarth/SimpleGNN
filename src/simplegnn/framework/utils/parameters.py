'''
Created on 04.12.2019

@author:
'''

import os

class Parameters(object):
    """
    Container for all experiment configuration parameters.

    This class holds all hyperparameters, paths, and settings required for running
    GNN experiments in the SimpleGNN framework. It organizes parameters into four
    main categories: benchmark configuration, evaluation settings, hyperparameters,
    and output control.

    Attributes
    ----------
    dropout : float or None
        Dropout probability for regularization. None if not set.

    Benchmark Graph Parameters
    --------------------------
    path : str
        Root directory path for dataset storage.
    results_path : str
        Directory path for saving experiment results and outputs.
    splits_path : str
        Directory path for train/validation/test split files.
    db : str
        Dataset name (e.g., 'MUTAG', 'ZINC', 'QM9').
    layers : list or None
        List of layer configurations for the neural network architecture.
    max_coding : int
        Maximum number of iterations for node label generation (default: 1).
    network_type : str
        Type of network architecture (default: "wl_1").
    node_features : int
        Dimensionality of node feature vectors. Initialized to max_coding.
    run_config : object or None
        RunConfiguration object containing full experiment configuration.
    use_features_as_channels : bool
        If True, treat input features as separate channels (default: False).
    batch_size : int
        Number of graphs per training batch. Set via set_data_param().
    load_splits : bool
        If True, load pre-existing train/val/test splits (default: True).

    Evaluation Parameters
    ---------------------
    run_id : int
        Unique identifier for this experimental run (default: 0).
    config_id : int
        Unique identifier for this hyperparameter configuration (default: 0).
    n_val_runs : int
        Number of validation runs for cross-validation (default: 10).
    validation_id : int
        Index of current validation fold (default: 0).
    balance_data : bool
        If True, use balanced sampling for imbalanced datasets (default: False).
    n_epochs : int
        Number of training epochs. Set via set_evaluation_param().
    convolution_grad : bool
        Enable gradient computation for convolutional layers.
    resize_grad : bool
        Enable gradient computation for graph resizing operations.

    Hyperparameters
    ---------------
    loss_function : str
        Name of loss function (e.g., 'CrossEntropyLoss', 'MSELoss').
    optimizer_function : str
        Name of optimizer (e.g., 'Adam', 'SGD').
    neural_net_layers : list of str
        Layer type specifications for network architecture (default: [""]).
    learning_rate : float
        Learning rate for optimizer (default: 0).
    optimizer : str
        Optimizer name, same as optimizer_function.

    Print and Draw Parameters
    --------------------------
    print_results : bool
        If True, print training/evaluation results to console (default: False).
    net_print_weights : bool
        If True, print network weight statistics (default: True).
    print_number : int
        Frequency of printing (every N batches/epochs) (default: 1).
    draw : bool
        If True, enable visualization/drawing features (default: False).
    draw_data : dict
        Storage for visualization data with 'X' and 'Y' keys (default: {'X': [], 'Y': []}).
    count : int
        Internal counter for tracking operations (default: 0).
    save_weights : bool
        If True, save model weights to disk (default: False).
    save_prediction_values : bool
        If True, save prediction outputs to file (default: False).
    plot_graphs : bool
        If True, generate and save plots (default: False).
    print_layer_init : bool
        If True, print layer initialization information (default: False).
    new_file_index : str
        Auto-generated file index for result files (default: '').
    save_file_name : str
        Filename for saving predictions. Set when save_prediction_values is True.

    Notes
    -----
    This class uses setter methods (set_data_param, set_evaluation_param, etc.)
    to configure groups of related parameters. All attributes are initialized in
    __init__ with default values and should be updated via the appropriate setter.

    The class is designed to be passed between different components of the
    framework (preprocessing, training, evaluation) to maintain configuration
    consistency.

    Examples
    --------
    >>> params = Parameters()
    >>> params.set_data_param(db='MUTAG', max_coding=3, layers=layer_config,
    ...                       node_features=16, run_config=config_obj)
    >>> params.set_evaluation_param(run_id=0, n_val_runs=10, validation_id=0,
    ...                             config_id=1, n_epochs=300, learning_rate=0.001,
    ...                             dropout=0.5, balance_data=False,
    ...                             convolution_grad=True, resize_graph=True)
    """

    def __init__(self):
        """
        Initialize Parameters with default values for all attributes.

        Sets up all experiment configuration parameters with safe defaults.
        All numeric values are initialized to 0 or None, booleans to False,
        strings to empty, and collections to empty lists/dicts.
        """

        self.dropout = None
        """
        BenchmarkGraphs parameters
        """
        self.path = ""
        self.results_path = ""
        self.splits_path = ""
        self.db = ""
        self.layers = None
        self.max_coding = 1
        self.network_type = "wl_1"
        self.node_features = self.max_coding
        self.run_config = None
        self.use_features_as_channels = False
        """
        Evaluation parameters
        """
        self.run_id = 0
        self.config_id = 0
        self.n_val_runs = 10
        self.validation_id = 0
        self.balance_data = False
        """
        Evaluation hyper parameters
        """
        self.loss_function = ""
        self.optimizer_function = ""
        self.neural_net_layers = [""]
        self.learning_rate = 0
        self.optimizer = ""
        """
        Print and draw parameters
        """
        self.print_results = False
        self.net_print_weights = True
        self.print_number = 1
        self.draw = False
        self.draw_data = {'X': [], 'Y': []}
        self.count = 0

        self.save_weights = False
        self.save_prediction_values = False
        self.plot_graphs = False
        self.print_layer_init = False

        self.new_file_index = ''

    def set_data_param(self, db, max_coding, layers, node_features, run_config):
        """
        Configure dataset and benchmark-related parameters.

        Sets paths, dataset name, layer configuration, and feature dimensions from
        the run configuration object. Extracts paths from run_config.config['paths']
        and batch size from run_config.config['batch_size'].

        Parameters
        ----------
        db : str
            Dataset name (e.g., 'MUTAG', 'ZINC', 'QM9').
        max_coding : int
            Maximum number of iterations for node label generation.
        layers : list
            List of layer configuration dictionaries defining network architecture.
        node_features : int
            Dimensionality of input node feature vectors.
        run_config : RunConfiguration
            Configuration object containing paths and batch_size in its config dict.

        Notes
        -----
        This method sets 10 attributes:
        - path, results_path, splits_path (extracted from run_config)
        - db, max_coding, layers, node_features (passed directly)
        - batch_size (extracted from run_config)
        - load_splits (extracted from run_config, defaults to True)
        - run_config (stores reference to full configuration)
        """
        self.path = run_config.config['paths']['data']
        self.results_path = run_config.config['paths']['results']
        self.splits_path = run_config.config['paths']['splits']
        self.db = db
        self.max_coding = max_coding
        self.layers = layers
        self.batch_size = run_config.config['batch_size']
        self.node_features = node_features
        self.load_splits = run_config.config.get('load_splits', True)
        self.run_config = run_config

    def set_evaluation_param(self, run_id, n_val_runs, validation_id, config_id, n_epochs, learning_rate, dropout, balance_data, convolution_grad, resize_graph):
        """
        Configure evaluation and training parameters.

        Sets experiment identifiers, cross-validation settings, training hyperparameters,
        and gradient computation flags.

        Parameters
        ----------
        run_id : int
            Unique identifier for this experimental run.
        n_val_runs : int
            Total number of validation runs for cross-validation.
        validation_id : int
            Index of the current validation fold (0 to n_val_runs-1).
        config_id : int
            Unique identifier for this hyperparameter configuration.
        n_epochs : int
            Number of training epochs.
        learning_rate : float
            Learning rate for the optimizer.
        dropout : float
            Dropout probability for regularization (0.0 to 1.0).
        balance_data : bool
            If True, use balanced sampling for imbalanced datasets.
        convolution_grad : bool
            If True, enable gradient computation for convolutional layers.
        resize_graph : bool
            If True, enable gradient computation for graph resizing operations.

        Notes
        -----
        The resize_graph parameter is stored as self.resize_grad (note the name change).
        This method sets 10 attributes that control the training and evaluation pipeline.
        """
        self.run_id = run_id
        self.n_val_runs = n_val_runs
        self.validation_id = validation_id
        self.config_id = config_id
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.balance_data = balance_data
        self.convolution_grad = convolution_grad
        self.resize_grad = resize_graph

    def set_hyper_param(self, learning_rate, loss_function, optimizer):
        """
        Configure core training hyperparameters.

        Sets the learning rate, loss function, and optimizer for model training.

        Parameters
        ----------
        learning_rate : float
            Learning rate for the optimizer.
        loss_function : str
            Name of the loss function (e.g., 'CrossEntropyLoss', 'MSELoss', 'L1Loss').
        optimizer : str
            Name of the optimizer (e.g., 'Adam', 'SGD', 'AdamW').

        Notes
        -----
        This method is typically called during hyperparameter grid search to set
        each configuration's training parameters.
        """
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.optimizer = optimizer

    def set_print_param(self, no_print, print_results, net_print_weights, print_number, draw, save_weights, save_prediction_values, plot_graphs, print_layer_init):
        """
        Configure output, logging, and visualization parameters.

        Controls what information is printed to console, saved to files, and visualized
        during training. The no_print flag acts as a master switch that disables all
        output when True.

        Parameters
        ----------
        no_print : bool
            Master switch. If True, disables ALL output/logging/visualization regardless
            of other parameter values. If False, uses individual parameter settings.
        print_results : bool
            If True, print training and evaluation results to console.
        net_print_weights : bool
            If True, print network weight statistics.
        print_number : int
            Frequency of printing (every N batches or epochs).
        draw : bool
            If True, enable drawing/visualization features.
        save_weights : bool
            If True, save model weights to disk.
        save_prediction_values : bool
            If True, save prediction outputs to file.
        plot_graphs : bool
            If True, generate and save plots.
        print_layer_init : bool
            If True, print layer initialization information.

        Notes
        -----
        When no_print=True, all 8 output flags are set to False (or 0 for print_number),
        overriding the values of the other parameters. This is useful for running large
        grid searches where verbose output would clutter logs.

        When no_print=False, each flag is set individually according to the corresponding
        parameter value.
        """
        if no_print:
            self.print_results = False
            self.net_print_weights = False
            self.print_number = 0
            self.draw = False
            self.save_weights = False
            self.save_prediction_values = False
            self.plot_graphs = False
            self.print_layer_init = False
        else:
            self.print_results = print_results
            self.net_print_weights = net_print_weights
            self.print_number = print_number
            self.draw = draw
            self.save_weights = save_weights
            self.save_prediction_values = save_prediction_values
            self.plot_graphs = plot_graphs
            self.print_layer_init = print_layer_init



    def save_predictions(self, output, labels):
        """
        Save model predictions and ground truth labels to file.

        Formats predictions and labels to 2 decimal places and appends them to
        the results file specified by self.save_file_name in self.results_path.

        Parameters
        ----------
        output : torch.Tensor
            Model predictions. Each element is extracted using .item() and rounded
            to 2 decimal places.
        labels : torch.Tensor
            Ground truth labels. Each element is extracted using .item() and formatted
            to 2 decimal places.

        Notes
        -----
        The output file format is:
                Labels: [label1, label2, ...]
            Prediction: [pred1, pred2, ...]

        The file is opened in append mode ('a'), so multiple calls accumulate results.
        Requires self.save_file_name and self.results_path to be set before calling.
        """
        out = [f"{round(x.item(), 2):.2f}" for x in output]
        lab = [f"{x.item():.2f}" for x in labels]

        with open(self.results_path + self.save_file_name, "a") as file_obj:
            file_obj.write("\n\t\t\tLabels: [{0}] \n\t\tPrediction: [{1}]\n\n".format(', '.join(map(str, lab)),
                                                                                      ', '.join(map(str, out))))

    def set_file_index(self, size):
        """
        Generate auto-incrementing file index for result files.

        Scans the results directory for existing .txt files, extracts numeric indices
        from filenames (assuming format: prefix_INDEX_suffix.txt), finds the maximum
        index, increments it, and formats it with zero-padding.

        Parameters
        ----------
        size : int
            Number of digits for zero-padding the index (e.g., size=3 produces '001', '042').

        Notes
        -----
        Algorithm:
        1. List all files in self.results_path
        2. For each .txt file, extract the second underscore-delimited field as an integer
        3. Find the maximum index (defaults to 0 if no valid indices found)
        4. Increment by 1
        5. Format with zero-padding to specified size

        The index is stored as a string in self.new_file_index.

        Example filename parsing:
        - "results_042_config1.txt" → index = 42
        - "results_001_final.txt" → index = 1

        If a filename doesn't match the expected pattern or the index field is not
        a valid integer, it is silently skipped (except clause).
        """
        # go through results folder and find the highest index at position two using _ as delimiter
        files = os.listdir(self.results_path)
        self.new_file_index = 0
        for file in files:
            if file.endswith('.txt'):
                try:
                    index = int(file.split('_')[1])
                    if index > self.new_file_index:
                        self.new_file_index = index
                except:
                    pass
        # increment the index if self.new_file_index is not 0
        self.new_file_index += 1
        # format the index to the length of the size
        self.new_file_index = str(self.new_file_index).zfill(size)
