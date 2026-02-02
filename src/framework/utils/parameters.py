'''
Created on 04.12.2019

@author:
'''

import os

class Parameters(object):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''

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
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.optimizer = optimizer

    def set_print_param(self, no_print, print_results, net_print_weights, print_number, draw, save_weights, save_prediction_values, plot_graphs, print_layer_init):
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
        out = [f"{round(x.item(), 2):.2f}" for x in output]
        lab = [f"{x.item():.2f}" for x in labels]

        with open(self.results_path + self.save_file_name, "a") as file_obj:
            file_obj.write("\n\t\t\tLabels: [{0}] \n\t\tPrediction: [{1}]\n\n".format(', '.join(map(str, lab)),
                                                                                      ', '.join(map(str, out))))

    def set_file_index(self, size):
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
