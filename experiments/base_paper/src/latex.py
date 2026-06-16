import json
from pathlib import Path

import numpy as np
import pandas as pd

from simplegnn.framework.core import FrameworkMain
from simplegnn.models.ShareGNN.layers.inv_based_message_passing import InvariantBasedMessagePassingLayer
from simplegnn.models.ShareGNN.layers.inv_based_pooling import InvariantBasedAggregationLayer
from simplegnn.framework.run_configuration import get_run_configs


def baseline_results(algorithm: str, datasets:list[str], path:str, sota:bool=False, first_column:str=''):
    '''
    get the results and print them in tabular form
    '''
    out_str = algorithm
    results = []
    # iterate over the datasets
    for dataset in datasets:
        dataset_path = f"{path}/{dataset}"
        # open and merge all csv files in dataset_path
        df_all = None
        for file in Path(dataset_path).iterdir():
            if file.suffix == '.csv':
                df = pd.read_csv(file, delimiter=";")
                # concatenate the dataframes
                if df_all is None:
                    df_all = df
                else:
                    df_all = pd.concat([df_all, df], ignore_index=True)
        # get groups from RunNumber and Algorithm
        df_results = []
        algorithm_groups = df_all.groupby('Algorithm')
        algorithm_mean_results = None
        for alg, algorithm_data in algorithm_groups:
            if alg == algorithm:
                run_group = algorithm_data.groupby('RunNumber')
                for run_number, group in run_group:
                    # remove algorithm and dataset columns
                    group = group.drop(columns=['Algorithm', 'Dataset'])
                    # multiply the TestAccuracy by 100
                    if not sota:
                        group['TestAccuracy'] = group['TestAccuracy'] * 100
                    else:
                        group['ValidationAccuracy'] = group['ValidationAccuracy'] * 100
                    # group each group by HyperparameterSVC and HyperparameterAlgo
                    grouped_results = group.groupby(['HyperparameterSVC', 'HyperparameterAlgo']).mean()
                    grouped_results_std = group.groupby(['HyperparameterSVC', 'HyperparameterAlgo']).std()
                    if not sota:
                        grouped_results['TestAccuracyStd'] = grouped_results_std['TestAccuracy']
                    else:
                        grouped_results['ValidationAccuracyStd'] = grouped_results_std['ValidationAccuracy']
                    grouped_results['RunNumber'] = run_number
                    if algorithm_mean_results is None:
                        algorithm_mean_results = grouped_results
                    else:
                        algorithm_mean_results = pd.concat([algorithm_mean_results, grouped_results], ignore_index=True)
                # get the mean of the algorithm_mean_results
                algorithm_mean_results = algorithm_mean_results.groupby(['RunNumber', 'HyperparameterSVC', 'HyperparameterAlgo']).mean()
                algorithm_mean_results['Dataset'] = dataset
                algorithm_mean_results['Algorithm'] = algorithm
                break
            else:
                continue

        # open summary_best_mean.csv
        # sort the algorithm_mean_results by ValidationAccuracy
        algorithm_mean_results = algorithm_mean_results.sort_values(by='ValidationAccuracy', ascending=False)
        data = algorithm_mean_results.iloc[0]
        if not sota:
            test_acc = data['TestAccuracy']
            test_std = data['TestAccuracyStd']
        else:
            test_acc = data['ValidationAccuracy']
            test_std = data['ValidationAccuracyStd']
        # round the test accuracy and test std to 2 decimal places
        test_acc = round(test_acc, 1)
        test_std = round(test_std, 1)
        results.append((test_acc, test_std))
        out_str += f' & ${test_acc} \\pm {test_std}$'


    return out_str, results


def share_gnn_results(algorithm: str, datasets:list[str], path:str, sota:bool=False):
    '''
    get the results and print them in tabular form
    '''
    out_str = algorithm
    results = []
    # iterate over the datasets
    for dataset in datasets:
        # open summary_best_mean.csv
        if not sota:
            df = pd.read_csv(f"{path}/{dataset}/summary_best_mean.csv", delimiter=",")
            test_acc = df['Test Accuracy Mean'].values[0]
            test_std = df['Test Accuracy Std'].values[0]
            # round the test accuracy and test std to 2 decimal places
            test_acc = round(test_acc, 1)
            test_std = round(test_std, 1)
            results.append((test_acc, test_std))
            out_str += f' & ${test_acc} \\pm {test_std}$'
            pass
        else:
            df = pd.read_csv(f"{path}/{dataset}/summary_sota.csv", delimiter=",")
            # get mean over configuration Id
            df = df.groupby('ConfigurationId').mean()
            # sort by Validation Accuracy
            df = df.sort_values(by='Validation Accuracy Mean', ascending=False)
            test_acc = df['Validation Accuracy Mean'].values[0]
            test_std = df['Validation Accuracy Std'].values[0]
            # round the test accuracy and test std to 2 decimal places
            test_acc = round(test_acc, 1)
            test_std = round(test_std, 1)
            results.append((test_acc, test_std))
            out_str += f' & ${test_acc} \\pm {test_std}$'
            pass
    return out_str, results

def fair_gnn_results(algorithm: str, datasets:list[str], path:str, first_column:str=''):
    '''
    get the results and print them in tabular form
    '''
    out_str = algorithm + first_column
    results = []
    # iterate over the datasets
    for dataset in datasets:
        if Path(f"{path}/{algorithm}_{dataset}_assessment/10_NESTED_CV/assessment_results.json").exists():
            with open(f"{path}/{algorithm}_{dataset}_assessment/10_NESTED_CV/assessment_results.json", 'r') as f:
                data = json.load(f)
                pass
            # open summary_best_mean.csv
            test_acc = data['avg_TS_score']
            test_std = data['std_TS_score']
            # round the test accuracy and test std to 2 decimal places
            test_acc = round(test_acc, 1)
            test_std = round(test_std, 1)
        elif Path(f"{path}/{algorithm}_{dataset}_assessment/5_NESTED_CV/assessment_results.json").exists():
            with open(f"{path}/{algorithm}_{dataset}_assessment/5_NESTED_CV/assessment_results.json", 'r') as f:
                data = json.load(f)
                pass
            # open summary_best_mean.csv
            test_acc = data['avg_TS_score']
            test_std = data['std_TS_score']
            # round the test accuracy and test std to 2 decimal places
            test_acc = round(test_acc, 1)
            test_std = round(test_std, 1)
        else:
            test_acc = 0
            test_std = 0
        results.append((test_acc, test_std))
        out_str += f' & ${test_acc} \\pm {test_std}$'
    return out_str, results

def print_table(first_columns, column_names, results, rules:list[int], with_colors:bool=True, with_std:bool=True, positive_negative_colors:bool=False):

    lines = []
    accuracies = results[:, :, :1].squeeze(2)
    stds = results[:, :, 1:].squeeze(2)

    # best three values per column
    best_column_values = np.array([np.sort(accuracies[:, i])[-3:] for i in range(accuracies.shape[1])])
    best_column_values = best_column_values.T

    if len(first_columns) != results.shape[0]:
        raise ValueError('The number of first columns and the number of rows in the results do not match')
    else:
        for i, first_column in enumerate(first_columns):
            line = first_column
            for j in range(accuracies.shape[1]):
                if accuracies[i, j] in best_column_values[:, j] and with_colors:
                    # get index of the value
                    index = np.where(best_column_values[:, j] == accuracies[i, j])[0][0]
                    if index == 0:
                        if with_std:
                            line += f' & \\ThirdColor{{{accuracies[i, j]} \\pm {stds[i, j]}}}'
                        else:
                            line += f' & \\ThirdColor{{{accuracies[i, j]}}}'
                    elif index == 1:
                        if with_std:
                            line += f' & \\SecondColor{{{accuracies[i, j]} \\pm {stds[i, j]}}}'
                        else:
                            line += f' & \\SecondColor{{{accuracies[i, j]}}}'
                    elif index == 2:
                        if with_std:
                            line += f' & \\FirstColor{{{accuracies[i, j]} \\pm {stds[i, j]}}}'
                        else:
                            line += f' & \\FirstColor{{{accuracies[i, j]}}}'
                else:
                    if positive_negative_colors:
                        if with_std:
                            if accuracies[i, j] > 0:
                                line += f' & \\FirstColor{{{accuracies[i, j]} \\pm {stds[i, j]}}}'
                            else:
                                line += f' & \\SecondColor{{{accuracies[i, j]} \\pm {stds[i, j]}}}'
                        else:
                            if accuracies[i, j] > 0:
                                line += f' & \\FirstColor{{{accuracies[i, j]}}}'
                            else:
                                line += f' & \\SecondColor{{{accuracies[i, j]}}}'
                    else:
                        if with_std:
                            line += f' & ${accuracies[i, j]} \\pm {stds[i, j]}$'
                        else:
                            line += f' & ${accuracies[i, j]}$'
            lines.append(line)
    # print the table in latex
    print('\\begin{tabular}{l' + 'c' * len(column_names) + '}')
    header = ' & '.join([''] + column_names) + '\\\\'
    print(header)
    print('\\toprule')
    for i,line in enumerate(lines):
        if i in rules:
            print('\\midrule')
        print(line + '\\\\')
    print('\\bottomrule')
    print('\\end{tabular}')

def fair_table():
    # print the large table
    datasets = ['NCI1', 'NCI109', 'Mutagenicity', 'DHFR', 'IMDB-BINARY', 'IMDB-MULTI']
    rows = []
    results = []

    baseline_algorithms = ['NoGKernel', 'WLKernel']
    baseline_first_columns = ['\\cite{Schulz2019OnTN}', '\\cite{DBLP:journals/jmlr/ShervashidzeSLMB11}']
    for i, algorithm in enumerate(baseline_algorithms):
        row_string, row_results = baseline_results(algorithm, datasets, 'results/base_paper/classification/RealWorld/Baseline/', first_column=baseline_first_columns[i])
        rows.append(row_string)
        results.append(row_results)

    fair_algorithms = ['GCN', 'GraphSAGE', 'GIN', 'GAT', 'GATv2']
    fair_first_columns = ['\\cite{DBLP:conf/iclr/KipfW17}', '\\cite{Hamilton2017InductiveRL}', '\\cite{DBLP:conf/iclr/XuHLJ19}', '\\cite{Velickovic2017GraphAN}', '\\cite{DBLP:conf/iclr/Brody0Y22}']
    fair_path = 'results/base_paper/classification/'
    for i, algorithm in enumerate(fair_algorithms):
        row_string, row_results = fair_gnn_results(algorithm, datasets, fair_path, first_column=fair_first_columns[i])
        rows.append(row_string)
        results.append(row_results)

    row, row_results = share_gnn_results('\\MyGNN (ours)', datasets, 'results/base_paper/classification/RealWorld/')
    rows.append(row)
    results.append(row_results)
    row, row_results = share_gnn_results('\\MyGNN-Random (ours)', datasets, 'results/base_paper/classification/RealWorld/Random/')
    rows.append(row)
    results.append(row_results)
    row, row_results = share_gnn_results('\\MyGNN-Encoder (ours)', datasets, 'results/base_paper/classification/RealWorld/Encoder/')
    rows.append(row)
    results.append(row_results)
    row, row_results = share_gnn_results('\\MyGNN-Decoder (ours)', datasets, 'results/base_paper/classification/RealWorld/Decoder/')
    rows.append(row)
    results.append(row_results)
    # results to numpy array
    results = np.array(results)

    first_columns = ([f'{x} {baseline_first_columns[i]}' for i, x in enumerate(baseline_algorithms)]
                     + [f'{x} {fair_first_columns[i]}' for i, x in enumerate(fair_algorithms)]
                     + ['\\textbf{\\MyGNN (ours)}', '\\textbf{\\MyGNN-Random (ours)}', '\\textbf{\\MyGNN-Encoder (ours)}', '\\textbf{\\MyGNN-Decoder (ours)}'])

    print_table(first_columns, ['\\textbf{NCI1}', '\\textbf{NCI109}', '\\textbf{Mutagenicity}', '\\textbf{DHFR}', '\\textbf{IMDB-B}', '\\textbf{IMDB-M}']
                , results,
                [2, 7])

def fair_table_full():
    # print the large table
    datasets = ['NCI1', 'NCI109', 'Mutagenicity', 'DHFR', 'IMDB-BINARY', 'IMDB-MULTI']
    rows = []
    results = []

    baseline_algorithms = ['NoGKernel', 'WLKernel']
    baseline_first_columns = ['\\cite{Schulz2019OnTN}', '\\cite{DBLP:journals/jmlr/ShervashidzeSLMB11}']
    for i, algorithm in enumerate(baseline_algorithms):
        row_string, row_results = baseline_results(algorithm, datasets, 'results/base_paper/classification/RealWorld/Baseline/', first_column=baseline_first_columns[i])
        rows.append(row_string)
        results.append(row_results)

    fair_algorithms = ['GCN', 'GraphSAGE', 'GIN', 'GAT', 'GATv2']
    fair_first_columns = ['\\cite{DBLP:conf/iclr/KipfW17}', '\\cite{Hamilton2017InductiveRL}', '\\cite{DBLP:conf/iclr/XuHLJ19}', '\\cite{Velickovic2017GraphAN}', '\\cite{DBLP:conf/iclr/Brody0Y22}']
    fair_path = 'results/base_paper/classification/'
    for i, algorithm in enumerate(fair_algorithms):
        row_string, row_results = fair_gnn_results(algorithm, datasets, fair_path, first_column=fair_first_columns[i])
        rows.append(row_string)
        results.append(row_results)

    row, row_results = share_gnn_results('\\MyGNN (ours)', datasets, 'results/base_paper/classification/RealWorld/')
    rows.append(row)
    results.append(row_results)
    row, row_results = share_gnn_results('\\MyGNN-Random (ours)', datasets, 'results/base_paper/classification/RealWorld/Random/')
    rows.append(row)
    results.append(row_results)
    row, row_results = share_gnn_results('\\MyGNN-Encoder (ours)', datasets, 'results/base_paper/classification/RealWorld/Encoder/')
    rows.append(row)
    results.append(row_results)
    row, row_results = share_gnn_results('\\MyGNN-Decoder (ours)', datasets, 'results/base_paper/classification/RealWorld/Decoder/')
    rows.append(row)
    results.append(row_results)
    # results to numpy array
    results = np.array(results)

    synthetic_datasets = ['LongRings100', 'EvenOddRingsCount16', 'EvenOddRings2_16', 'CSL', 'Snowflakes']
    synthetic_rows = []
    synthetic_results = []
    for i, algorithm in enumerate(baseline_algorithms):
        row_string, row_results = baseline_results(algorithm, synthetic_datasets, 'results/base_paper/classification/Synthetic/Baseline/', first_column=baseline_first_columns[i])
        synthetic_rows.append(row_string)
        synthetic_results.append(row_results)

    for i, algorithm in enumerate(fair_algorithms):
        row_string, row_results = fair_gnn_results(algorithm, synthetic_datasets, fair_path, first_column=fair_first_columns[i])
        synthetic_rows.append(row_string)
        synthetic_results.append(row_results)

    row, row_results = share_gnn_results('\\MyGNN (ours)', synthetic_datasets, 'results/base_paper/classification/Synthetic/')
    synthetic_rows.append(row)
    synthetic_results.append(row_results)
    row, row_results = share_gnn_results('\\MyGNN-Random (ours)', synthetic_datasets, 'results/base_paper/classification/Synthetic/Random/')
    synthetic_rows.append(row)
    synthetic_results.append(row_results)
    row, row_results = share_gnn_results('\\MyGNN-Encoder (ours)', synthetic_datasets, 'results/base_paper/classification/Synthetic/Encoder/')
    synthetic_rows.append(row)
    synthetic_results.append(row_results)
    row, row_results = share_gnn_results('\\MyGNN-Decoder (ours)', synthetic_datasets, 'results/base_paper/classification/Synthetic/Decoder/')
    synthetic_rows.append(row)
    synthetic_results.append(row_results)
    # results to numpy array
    synthetic_results = np.array(synthetic_results)
    # concatenate the results
    results = np.concatenate((results, synthetic_results), axis=1)


    first_columns = ([f'{x} {baseline_first_columns[i]}' for i, x in enumerate(baseline_algorithms)]
                     + [f'{x} {fair_first_columns[i]}' for i, x in enumerate(fair_algorithms)]
                     + ['\\textbf{\\MyGNN (ours)}', '\\textbf{\\MyGNN-Random (ours)}', '\\textbf{\\MyGNN-Encoder (ours)}', '\\textbf{\\MyGNN-Decoder (ours)}'])

    print_table(first_columns, ['\\textbf{NCI1}', '\\textbf{NCI109}', '\\textbf{Mutagen.}', '\\textbf{DHFR}', '\\textbf{IMDB-B}', '\\textbf{IMDB-M}',
                                '\\textbf{RingT1}', '\\textbf{RingT2}', '\\textbf{RingT3}', '\\textbf{CSL}', '\\textbf{Snowfl.}']
                , results,
                [2, 7])



def sota_baseline_and_share():
    datasets = ['NCI1', 'NCI109', 'IMDB-BINARY', 'IMDB-MULTI']
    rows = []
    results = []

    baseline_algorithms = ['NoGKernel', 'WLKernel']
    baseline_first_columns = ['\\cite{Schulz2019OnTN}', '\\cite{DBLP:journals/jmlr/ShervashidzeSLMB11}']
    for i, algorithm in enumerate(baseline_algorithms):
        row_string, row_results = baseline_results(algorithm, datasets, 'results/base_paper/classification/Sota/Baseline/', first_column=baseline_first_columns[i], sota=True)
        rows.append(row_string)
        results.append(row_results)

    row, row_results = share_gnn_results('\\MyGNN (ours)', datasets, 'results/base_paper/classification/Sota/', sota=True)
    rows.append(row)
    results.append(row_results)
    row, row_results = share_gnn_results('\\MyGNN-Random (ours)', datasets, 'results/base_paper/classification/Sota/Random/', sota=True)
    rows.append(row)
    results.append(row_results)
    first_columns = ([f'{x} {baseline_first_columns[i]}' for i, x in enumerate(baseline_algorithms)]
                     + ['\\textbf{\\MyGNN}', '\\textbf{\\MyGNN-Random}'])
    print_table(first_columns, ['\\textbf{NCI1}', '\\textbf{NCI109}', '\\textbf{IMDB-B}', '\\textbf{IMDB-M}']
                , np.array(results),
                [], with_colors=False)


def synthetic_table():
    datasets = ['CSL', 'EvenOddRings2_16', 'EvenOddRingsCount16', 'LongRings100', 'Snowflakes']
    rows = []
    results = []

    baseline_algorithms = ['NoGKernel', 'WLKernel']
    baseline_first_columns = ['\\cite{Schulz2019OnTN}', '\\cite{DBLP:journals/jmlr/ShervashidzeSLMB11}']
    for i, algorithm in enumerate(baseline_algorithms):
        row_string, row_results = baseline_results(algorithm, datasets, 'results/base_paper/classification/Synthetic/Baseline/', first_column=baseline_first_columns[i])
        rows.append(row_string)
        results.append(row_results)

    fair_algorithms = ['GCN', 'GraphSAGE', 'GIN', 'GAT', 'GATv2']
    fair_first_columns = ['\\cite{DBLP:conf/iclr/KipfW17}', '\\cite{Hamilton2017InductiveRL}', '\\cite{DBLP:conf/iclr/XuHLJ19}', '\\cite{Velickovic2017GraphAN}', '\\cite{DBLP:conf/iclr/Brody0Y22}']
    fair_path = 'results/base_paper/classification/'
    for i, algorithm in enumerate(fair_algorithms):
        row_string, row_results = fair_gnn_results(algorithm, datasets, fair_path, first_column=fair_first_columns[i])
        rows.append(row_string)
        results.append(row_results)

    row, row_results = share_gnn_results('\\MyGNN', datasets, 'results/base_paper/classification/Synthetic/')
    rows.append(row)
    results.append(row_results)
    row, row_results = share_gnn_results('\\MyGNN-Random', datasets, 'results/base_paper/classification/Synthetic/Random/')
    rows.append(row)
    results.append(row_results)
    row, row_results = share_gnn_results('\\MyGNN-Encoder', datasets, 'results/base_paper/classification/Synthetic/Encoder/')
    rows.append(row)
    results.append(row_results)
    row, row_results = share_gnn_results('\\MyGNN-Decoder', datasets, 'results/base_paper/classification/Synthetic/Decoder/')
    rows.append(row)
    results.append(row_results)

    first_columns = ([f'{x} {baseline_first_columns[i]}' for i, x in enumerate(baseline_algorithms)]
                        + [f'{x} {fair_first_columns[i]}' for i, x in enumerate(fair_algorithms)]
                        + ['\\textbf{\\MyGNN (ours)}', '\\textbf{\\MyGNN-Random (ours)}', '\\textbf{\\MyGNN-Encoder (ours)}', '\\textbf{\\MyGNN-Decoder (ours)}'])
    print_table(first_columns, ['\\textbf{CSL}', '\\textbf{EvenOddRings2}', '\\textbf{EvenOddRingsCount}', '\\textbf{LongRings}', '\\textbf{Snowflakes}']
                , np.array(results),
                [2, 7])


def features_evaluation():
    fair_algorithms = ['GCN', 'GraphSAGE', 'GIN', 'GAT', 'GATv2']
    fair_first_columns = ['\\cite{DBLP:conf/iclr/KipfW17}', '\\cite{Hamilton2017InductiveRL}',
                          '\\cite{DBLP:conf/iclr/XuHLJ19}', '\\cite{Velickovic2017GraphAN}',
                          '\\cite{DBLP:conf/iclr/Brody0Y22}']

    datasets = ['NCI1', 'DHFR', 'IMDB-BINARY', 'IMDB-MULTI']
    datasets_features = ['NCI1Features', 'DHFRFeatures', 'IMDB-BINARYFeatures', 'IMDB-MULTIFeatures']
    rows = []
    features_results = []
    results = []
    fair_path = 'results/base_paper/classification/'

    for i, algorithm in enumerate(fair_algorithms):
        row_string, row_results = fair_gnn_results(algorithm, datasets, fair_path,
                                                   first_column=fair_first_columns[i])
        rows.append(row_string)
        results.append(row_results)

    for i, algorithm in enumerate(fair_algorithms):
        row_string, row_results = fair_gnn_results(algorithm, datasets_features, fair_path,
                                                   first_column=fair_first_columns[i])
        rows.append(row_string)
        features_results.append(row_results)

    results = np.array(results)
    features_results = np.array(features_results)
    results = features_results - results
    # round the results to 1 decimal place
    results = np.round(results, 1)

    first_columns = [f'{x} {fair_first_columns[i]}' for i, x in enumerate(fair_algorithms)]
    print_table(first_columns, ['\\textbf{NCI1}', '\\textbf{DHFR}', '\\textbf{IMDB-B}', '\\textbf{IMDB-M}']
                , np.array(results),
                [], with_colors=False, with_std=False, positive_negative_colors=True)



def labels_to_string(label_string:str):
    # if label string of type wl_x return WL, Iterations x
    # wl_labeled_x return WL with Node Labels, Iterations x
    # simple_cycles_x return Pattern: Simple Cycles, Max.~Length x
    if 'wl' in label_string:
        if not 'primary' in label_string:
            return 'WL, Iterations ' + label_string.split('_')[-1]
        else:
            # remove _primary
            label_string = label_string.replace('_primary', '')
            return 'WL + Node Labels, Iterations ' + label_string.split('_')[-1]
    elif 'wl_labeled' in label_string:
        return 'WL with Node Labels, Iterations ' + label_string.split('_')[-1]
    elif 'simple_cycles' in label_string:
        if not 'primary' in label_string:
            return 'Simple Cycles, Max.~Length ' + label_string.split('_')[-1]
        else:
            # remove _primary
            label_string = label_string.replace('_primary', '')
            return f'Simple Cycles, Max.~Length {label_string.split("_")[-1]} + Node Labels'
    elif 'induced_cycles' in label_string:
        if not 'primary' in label_string:
            return 'Induced Cycles, Max.~Length ' + label_string.split('_')[-1]
        else:
            # remove _primary
            label_string = label_string.replace('_primary', '')
            return f'Induced Cycles, Max.~Length {label_string.split("_")[-1]} + Node Labels'
    elif 'clique' in label_string:
        if not 'primary' in label_string:
            return 'Clique, Max.~Size ' + label_string.split('_')[-1]
        else:
            # remove _primary
            label_string = label_string.replace('_primary', '')
            return f'Clique, Max.~Size {label_string.split("_")[-1]} + Node Labels'
    elif 'subgraph_0' in label_string:
        if not 'primary' in label_string:
            return 'Clique Size 4'
        else:
            return 'Clique Size 4 + Node Labels,'
    elif 'subgraph_1' in label_string:
        if not 'primary' in label_string:
            return 'Triangles + Node Degree'
        else:
            return 'Triangles + Node Degree + Node Labels'
    elif 'subgraph_2' in label_string:
        if not 'primary' in label_string:
            return 'Squares + Node Degree'
        else:
            return 'Squares + Node Degree + Node Labels'
    elif 'subgraph_3' in label_string:
        if not 'primary' in label_string:
            return 'Triangles, Squares + Node Degree'
        else:
            return 'Triangles, Squares + Node Degree + Node Labels'

    elif 'primary' in label_string:
        return 'Node Labels'

def properties_to_string(properties_dict):
    # if there are consecutive numbers in properties_dict use , \\ldots between min and max
    min_val = min(properties_dict['values'])
    max_val = max(properties_dict['values'])
    if len(properties_dict['values']) > 3 and max_val - min_val == len(properties_dict['values']) - 1:
        return f'{min_val}, \\ldots, {max_val}'
    properties_string = ''
    for i,x in enumerate(properties_dict['values']):
        if i == len(properties_dict['values']) - 1:
            properties_string += f'{x}'
        else:
            properties_string += f'{x}, '
    return properties_string

def format_number(number:str):
    # add \, to each 3rd digit
    for i in range(len(number) - 3, 0, -3):
        number = number[:i] + '\\,' + number[i:]
    return number



def training_and_preprocessing_time(share_gnn_type=''):
    datasets_real_world = ['NCI1', 'NCI109', 'Mutagenicity', 'DHFR', 'IMDB-BINARY', 'IMDB-MULTI']
    dataset_synthetic = ['LongRings100', 'EvenOddRingsCount16', 'EvenOddRings2_16', 'CSL', 'Snowflakes']
    results = {key: dict() for key in datasets_real_world + dataset_synthetic}

    # check if results file already exists
    appendix = ''
    if share_gnn_type != '':
        appendix = f'_{share_gnn_type}'
    if Path(f'results/base_paper/classification/Latex/training_preprocessing_time{appendix}.json').exists():
        results = json.load(open(f'results/base_paper/classification/Latex/training_preprocessing_time{appendix}.json', 'r'))
    else:

        path_real_world = f'results/base_paper/classification/RealWorld/{share_gnn_type}'
        path_synthetic = f'results/base_paper/classification/Synthetic/{share_gnn_type}'
        for path, datasets in [[path_real_world, datasets_real_world], [path_synthetic, dataset_synthetic]]:
            if Path(path).exists():
                # get number of parameters
                for dataset in datasets:
                    # get the file from results folder that contains Best and Network
                    for file in Path(f'{path}/{dataset}/Results').iterdir():
                        if 'Best_Configuration' in file.name and 'Network' in file.name:
                            with open(file, 'r') as f:
                                data = f.read()
                                data = data.split('\n')
                                for line in data:
                                    if 'Trainable Parameters' in line:
                                        if 'trainable_parameters_per_layer' in results[dataset]:
                                            results[dataset]['trainable_parameters_per_layer'].append(int(line.split(':')[-1].strip()))
                                        else:
                                            results[dataset]['trainable_parameters_per_layer'] = [int(line.split(':')[-1].strip())]
                                    if 'Total trainable parameters' in line:
                                        num_parameters = int(line.split(':')[-1].strip())
                                        results[dataset]['parameters'] = num_parameters
                                        break
                            break
                # get avg best epoch
                for dataset in datasets:
                        # get the file from results folder that contains Best and Network
                        df = pd.read_csv(f'{path}/{dataset}/summary_best_mean.csv', delimiter=",")
                        results[dataset]['mean_epoch'] = df['Epoch Mean'].values[0] + 1
                        results[dataset]['std_epoch'] = df['Epoch Std'].values[0]

                # get avg best epoch and avg epoch runtime
                for dataset in datasets:
                    # get the file from results folder that contains Best and Network
                    df_all = None
                    for file in Path(f'{path}/{dataset}/Results').iterdir():
                        if f'{dataset}_Best_Configuration' in file.name and '.csv' in file.suffix:
                            df = pd.read_csv(file, delimiter=";")
                            # concatenate the dataframes
                            if df_all is None:
                                df_all = df
                            else:
                                df_all = pd.concat([df_all, df], ignore_index=True)
                    # get the mean of all epoch times
                    mean_epoch_time = df_all['EpochTime'].mean()
                    std_epoch_time = df_all['EpochTime'].std()
                    results[dataset]['mean_epoch_time'] = mean_epoch_time
                    results[dataset]['std_epoch_time'] = std_epoch_time

                # get preprocessing time for distances
                with open(Path(path).joinpath('generation_times_properties.txt'), 'r') as f:
                    for i, line in enumerate(f):
                        if i != 0:
                            dataset, _ , time = line.split(',')
                            results[dataset.strip()]['preprocessing_time'] = float(time.strip())



                # get preprocessing time for features
                if share_gnn_type == '':
                    preprocessing_times = dict()
                    with open(Path(path).joinpath('generation_times_labels.txt'), 'r') as f:
                        for i, line in enumerate(f):
                            if i != 0:
                                dataset, label , time = line.split(',')
                                # remove _None from label
                                label = label.replace('_None', '')
                                # if the word simple or induced appears twice, remove all after the second appearance
                                if label.count('simple') > 1:
                                    label = label[:label.rfind('simple')]
                                if label.count('induced') > 1:
                                    label = label[:label.rfind('induced')]

                                if dataset not in preprocessing_times:
                                    preprocessing_times[dataset.strip()] = dict()
                                    preprocessing_times[dataset.strip()]['all'] = 0
                                preprocessing_times[dataset.strip()][label.strip()] = float(time.strip())
                                preprocessing_times[dataset.strip()]['all'] += float(time.strip())
                    for dataset in preprocessing_times:
                        if dataset in results:
                            results[dataset]['preprocessing_times'] = preprocessing_times[dataset]
                        else:
                            results[dataset] = dict()
                            results[dataset]['preprocessing_times'] = preprocessing_times[dataset]

                # best label strings per dataset
                ## Real World Data
                if share_gnn_type == '':
                    config_path = Path('experiments/base_paper/classification/configs/main_config_fair_real_world.yml')
                elif share_gnn_type == 'Random':
                    config_path = Path('experiments/base_paper/classification/configs/main_config_fair_real_world_random_variation.yml')
                else:
                    raise ValueError('share_gnn_type not recognized')
                if path == path_synthetic:
                    if share_gnn_type == '':
                        config_path = Path('experiments/base_paper/classification/configs/main_config_fair_synthetic.yml')
                    elif share_gnn_type == 'Random':
                        config_path = Path('experiments/base_paper/classification/configs/main_config_fair_synthetic_random_variation.yml')
                    else:
                        raise ValueError('share_gnn_type not recognized')
                experiment = FrameworkMain(Path(config_path))
                experiment.preprocessing(num_threads=1)


                for dataset in datasets:
                    print(f'Load model for Dataset: {dataset}')
                    net = experiment.load_model(f'{dataset}', 0, 0, 0, best=True)
                    print('Loading Finished')
                    label_strings = set()
                    for i, layer in enumerate(net.net_layers):
                        if isinstance(layer, InvariantBasedMessagePassingLayer):
                            if 'heads' in results[dataset]:
                                results[dataset]['heads'].append(len(layer.n_source_labels))
                            else:
                                results[dataset]['heads'] = [len(layer.n_source_labels)]
                            if 'property_names' in results[dataset]:
                                results[dataset]['property_names'].append(layer.property_descriptions)
                            else:
                                results[dataset]['property_names'] = [layer.property_descriptions]
                            if 'property_dicts' in results[dataset]:
                                results[dataset]['property_dicts'].append([x.property_dict for x in net.para.layers[i].layer_heads])
                            else:
                                results[dataset]['property_dicts'] = [[x.property_dict for x in net.para.layers[i].layer_heads]]
                            for x in layer.source_label_descriptions:
                                label_strings.add(x)
                                if 'layer_labels_head' in results[dataset]:
                                    results[dataset]['layer_labels_head'].append(x)
                                else:
                                    results[dataset]['layer_labels_head'] = [x]
                            for x in layer.target_label_descriptions:
                                label_strings.add(x)
                                if 'layer_labels_tail' in results[dataset]:
                                    results[dataset]['layer_labels_tail'].append(x)
                                else:
                                    results[dataset]['layer_labels_tail'] = [x]
                            for x in layer.bias_label_descriptions:
                                label_strings.add(x)
                                if 'layer_labels_bias' in results[dataset]:
                                    results[dataset]['layer_labels_bias'].append(x)
                                else:
                                    results[dataset]['layer_labels_bias'] = [x]
                        elif isinstance(layer, InvariantBasedAggregationLayer):
                            for x in layer.source_label_descriptions:
                                label_strings.add(x)
                                if 'layer_labels_aggregation' in results[dataset]:
                                    results[dataset]['layer_labels_aggregation'].append(x)
                                else:
                                    results[dataset]['layer_labels_aggregation'] = [x]
                        else:
                            pass
                    # remove all strings with _primary from label_strings
                    label_strings = set([x for x in label_strings if '_primary' not in x])
                    if share_gnn_type == '':
                        results[dataset]['preprocessing_time_labels'] = sum([preprocessing_times[dataset][label] for label in label_strings])

        # save results in file
        with open(f'results/base_paper/classification/Latex/training_preprocessing_time{appendix}.json', 'w') as f:
            json.dump(results, f)


    ### Best Run Table
    dataset_synthetic_names = ['RingTransfer1', 'RingTransfer2', 'RingTransfer3', 'CSL', 'Snowflakes']
    # create a table with the results Best Epoch, Epoch Time (s), #Parameters for the Best parameter configuration
    table_str = '\\begin{tabular}{lcc|ccr|cr|r}\n'
    table_str += '\\toprule\n'
    table_str += 'Dataset & Best Epoch & Time per Epoch (s) & \\multicolumn{3}{c}{Encoder Layers} & \\multicolumn{2}{c}{Decoder Layer} & \\# Total Parameters \\\\ \n'
    table_str += ' & & & Invariants & Distances & \\# Parameters & Invariants & \\# Parameters & \\\\ \n'
    table_str += '\\midrule\n'
    for dataset in datasets_real_world:
        table_str += f'{dataset} & ${round(results[dataset]["mean_epoch"],1)} \\pm {round(results[dataset]["std_epoch"],1)}$ & ${round(results[dataset]["mean_epoch_time"],1)} \\pm {round(results[dataset]["std_epoch_time"],1)}$ & {labels_to_string(results[dataset]["layer_labels_head"][0])} & {properties_to_string(results[dataset]["property_dicts"][0][0])} & {format_number(str(results[dataset]["trainable_parameters_per_layer"][0]))} & {labels_to_string(results[dataset]["layer_labels_aggregation"][0])} & {format_number(str(results[dataset]["trainable_parameters_per_layer"][-1]))} & ${format_number(str(results[dataset]["parameters"]))}$ \\\\ \n'
    table_str += '\\midrule\n'
    for i, dataset in enumerate(dataset_synthetic):
        table_str += f'{dataset_synthetic_names[i]} & ${round(results[dataset]["mean_epoch"],1)} \\pm {round(results[dataset]["std_epoch"],1)}$ & ${round(results[dataset]["mean_epoch_time"],1)} \\pm {round(results[dataset]["std_epoch_time"],1)}$ & {labels_to_string(results[dataset]["layer_labels_head"][0])} & {properties_to_string(results[dataset]["property_dicts"][0][0])} & {format_number(str(results[dataset]["trainable_parameters_per_layer"][0]))} & {labels_to_string(results[dataset]["layer_labels_aggregation"][0])} & {format_number(str(results[dataset]["trainable_parameters_per_layer"][-1]))} & ${format_number(str(results[dataset]["parameters"]))}$ \\\\ \n'
    table_str += '\\bottomrule\n'
    table_str += '\\end{tabular}\n'
    # save table under best run properties table
    with open(f'results/base_paper/classification/Latex/best_run_details_table{appendix}.txt', 'w') as f:
        f.write(table_str)

    if share_gnn_type == '':
        ### Preprocessing Time Table
        # create a table with the results Best Epoch, Epoch Time (s), #Parameters for the Best parameter configuration
        table_str = '\\begin{tabular}{lrr}\n'
        table_str += '\\toprule\n'
        table_str += 'Dataset & Preprocessing Distances (s) & Preprocessing Labels (s) \\\\ \n'
        table_str += '\\midrule\n'
        for dataset in datasets_real_world:
            table_str += f'{dataset} & ${round(results[dataset]["preprocessing_time"],1)}$ & ${round(results[dataset]["preprocessing_times"]["all"],1)}$ \\\\ \n'
        table_str += '\\midrule\n'
        for i, dataset in enumerate(dataset_synthetic):
            table_str += f'{dataset_synthetic_names[i]} & ${round(results[dataset]["preprocessing_time"],1)}$ & ${round(results[dataset]["preprocessing_times"]["all"],1)}$ \\\\ \n'
        table_str += '\\bottomrule\n'
        table_str += '\\end{tabular}\n'
        # save table under best run properties table
        with open('results/base_paper/classification/Latex/preprocessing_times.txt', 'w') as f:
            f.write(table_str)



def hyper_parameter_configurations():
    # check if json has been produced
    json_files = ['results/base_paper/classification/Latex/molecule_convolution_configs.json',
                    'results/base_paper/classification/Latex/molecule_aggregation_configs.json',
                    'results/base_paper/classification/Latex/social_convolution_configs.json',
                    'results/base_paper/classification/Latex/social_aggregation_configs.json']
    if all([Path(x).exists() for x in json_files]):
        pass
    else:
        molecule = 'NCI1'
        social = 'IMDB-BINARY'
        # best label strings per dataset
        ## Real World Data
        config_path = Path('experiments/base_paper/classification/configs/main_config_fair_real_world.yml')
        experiment = FrameworkMain(Path(config_path))
        experiment.preprocessing(num_threads=1)
        run_configs_molecule = get_run_configs(experiment.network_configurations[molecule][0])
        run_configs_social = get_run_configs(experiment.network_configurations[social][0])
        for run_configs in [run_configs_molecule, run_configs_social]:
            convolution_configs = set()
            aggregation_configs = set()
            for run_config in run_configs:
                for layer in run_config.layers:
                    if layer.layer_type == 'convolution':
                        layer_string = layer.get_layer_label_strings()
                        for x in layer_string:
                            convolution_configs.add(labels_to_string(x))
                    elif layer.layer_type == 'aggregation':
                        layer_string = layer.get_layer_label_strings()
                        for x in layer_string:
                            aggregation_configs.add(labels_to_string(x))
            convolution_configs = sorted(list(convolution_configs))
            aggregation_configs = sorted(list(aggregation_configs))
            # save the configurations in a file
            if run_configs == run_configs_molecule:
                with open(f'results/base_paper/classification/Latex/molecule_convolution_configs.json', 'w') as f:
                    json.dump(list(convolution_configs), f)
                with open(f'results/base_paper/classification/Latex/molecule_aggregation_configs.json', 'w') as f:
                    json.dump(list(aggregation_configs), f)
            elif run_configs == run_configs_social:
                with open(f'results/base_paper/classification/Latex/social_convolution_configs.json', 'w') as f:
                    json.dump(list(convolution_configs), f)
                with open(f'results/base_paper/classification/Latex/social_aggregation_configs.json', 'w') as f:
                    json.dump(list(aggregation_configs), f)
    molecule_convolution_configs = json.load(
        open('results/base_paper/classification/Latex/molecule_convolution_configs.json', 'r'))
    molecule_aggregation_configs = json.load(
        open('results/base_paper/classification/Latex/molecule_aggregation_configs.json', 'r'))
    social_convolution_configs = json.load(open('results/base_paper/classification/Latex/social_convolution_configs.json', 'r'))
    social_aggregation_configs = json.load(open('results/base_paper/classification/Latex/social_aggregation_configs.json', 'r'))

    # create four tables: Molecules Encoder Invariants, Molecules Decoder Invariants, Social Encoder Invariants, Social Decoder Invariants
    for x, y, z in zip([molecule_convolution_configs, molecule_aggregation_configs, social_convolution_configs, social_aggregation_configs],
                 ['Molecules Encoder Invariants', 'Molecules Decoder Invariants', 'Social Encoder Invariants', 'Social Decoder Invariants'],
                    ['molecule_convolution_configs', 'molecule_aggregation_configs', 'social_convolution_configs', 'social_aggregation_configs']):

        table_string = '\\begin{tabular}{c}\n'
        table_string += '\\toprule\n'
        table_string += f'{y} \\\\ \n'
        table_string += '\\midrule\n'
        for config in x:
            table_string += f'{config} \\\\ \n'
        table_string += '\\bottomrule\n'
        table_string += '\\end{tabular}\n'
        with open(f'results/base_paper/classification/Latex/{z}.txt', 'w') as f:
            f.write(table_string)

def main():
    # create Latex dir under Results
    Path('results/base_paper/classification/Latex').mkdir(parents=True, exist_ok=True)
    hyper_parameter_configurations()
    training_and_preprocessing_time()
    training_and_preprocessing_time('Random')
    fair_table_full()
    print('\n\n\n\n')
    sota_baseline_and_share()
    print('\n\n\n\n')
    features_evaluation()
    synthetic_table()


if __name__ == '__main__':
    main()