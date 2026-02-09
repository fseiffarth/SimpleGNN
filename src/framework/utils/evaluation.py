import os
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from utils.utils import is_pruning


def epoch_accuracy(db_name, y_val, ids):
    if y_val == 'Train':
        y_val = 'EpochAccuracy'
        size = 'TrainingSize'
    elif y_val == 'Validation':
        y_val = 'ValidationAccuracy'
        size = 'ValidationSize'
    elif y_val == 'Test':
        y_val = 'TestAccuracy'
        size = 'TestSize'

    # load the data from Results/{db_name}/Results/{db_name}_{id_str}_Results_run_id_{run_id}.csv as a pandas dataframe for all run_ids in the directory
    # ge all those files
    files = []
    network_files = []
    for file in os.listdir(f"Results/{db_name}/Results"):
        for id in ids:
            id_str = str(id).zfill(6)
            # file contains the id_str
            if id_str in file:
                if file.startswith(f"{db_name}_") and file.endswith(".csv"):
                    files.append(file)
                elif file.startswith(f"{db_name}_") and file.endswith(".txt"):
                    network_files.append(file)
    df_all = None
    for i, file in enumerate(files):
        # get file id
        file_id = file.split('_')[1]
        df = pd.read_csv(f"Results/{db_name}/Results/{file}", delimiter=";")
        # add the file id to the dataframe
        df['FileId'] = file_id
        # concatenate the dataframes
        if df_all is None:
            df_all = df
        else:
            df_all = pd.concat([df_all, df], ignore_index=True)
    # open network file and read the network
    network_legend = {}
    for i, file in enumerate(network_files):
        with open(f"Results/{db_name}/Results/{file}", "r") as f:
            # get first line
            line = f.readline()
            # get string between [ and ]
            line = line[line.find("[") + 1:line.find("]")]
            simple = True
            if simple:
                # split by ,
                line = line.split(", ")
                # join the strings
                id = file.split('_')[1]
                network_legend[id] = f'Id:{id}, {"".join(line)}'
            else:
                # split by , not in ''
                line = line.split(", ")
                k = line[1].split("_")[1].split(":")[0]
                d = 0
                if ":" in line[1]:
                    d = len(line[1].split("_")[1].split(":")[1].split(","))
                bound = line[0]
                # cound number of occurrences of "wl" in line
                L = sum([1 for i in line if "wl" in i]) - 1

                # remove last element
                line = line[:-1]
                # join the strings with ;
                line = ";".join(line)
                id = file.split('_')[1]
                # remove ' from k,d,bound and L
                k = k.replace("'", "")
                bound = bound.replace("'", "")
                if d == 0:
                    # replace d and bound by '-'
                    d = '-'
                    bound = '-'
                if k == '20':
                    k = 'max'
                # network_legend[id] = f'Id:{id}, {line}'
                char = '\u00b2'
                if L == 0:
                    char = ''
                elif L == 1:
                    char = '\u00b9'
                elif L == 2:
                    char = '\u00b2'
                elif L == 3:
                    char = '\u00b3'
                network_legend[id] = f'({k},{d},{bound}){char}'

    # group by file id
    groups = df_all.groupby('FileId')
    # for each group group by epoch and get the mean and std
    epoch_loss_column_name = None
    for col in df_all.columns:
        if 'EpochLoss' in col:
            epoch_loss_column_name = col
            break
    if epoch_loss_column_name is None:
        print("No column found that contains 'EpochLoss'")
        return
    for i, id in enumerate(ids):
        id_str = str(id).zfill(6)
        group = groups.get_group(id_str).copy()
        group['EpochAccuracy'] = group['EpochAccuracy'] * group['TrainingSize']
        group[epoch_loss_column_name] *= group['TrainingSize']
        group['ValidationAccuracy'] *= group['ValidationSize']
        group['ValidationLoss'] *= group['ValidationSize']
        group['TestAccuracy'] *= group['TestSize']
        # f = lambda x: sum(x['TrainingSize'] * x['EpochAccuracy']) / sum(x['TrainingSize'])

        group_mean = group.groupby('Epoch').mean(numeric_only=True)
        group_mean['EpochAccuracy'] /= group_mean['TrainingSize']
        group_mean[epoch_loss_column_name] /= group_mean['TrainingSize']
        group_mean['ValidationAccuracy'] /= group_mean['ValidationSize']
        group_mean['ValidationLoss'] /= group_mean['ValidationSize']
        group_mean['TestAccuracy'] /= group_mean['TestSize']
        group_std = group.groupby('Epoch').std(numeric_only=True)
        group_std['EpochAccuracy'] /= group_mean['TrainingSize']
        group_std[epoch_loss_column_name] /= group_mean['TrainingSize']
        group_std['ValidationAccuracy'] /= group_mean['ValidationSize']
        group_std['ValidationLoss'] /= group_mean['ValidationSize']
        group_std['TestAccuracy'] /= group_mean['TestSize']
        # plot the EpochAccuracy vs Epoch
        if i == 0:
            ax = group_mean.plot(y=y_val, yerr=group_std[y_val], label=network_legend[id_str])
        else:
            group_mean.plot(y=y_val, yerr=group_std[y_val], ax=ax, label=network_legend[id_str])

    # save to tikz
    # tikzplotlib.save(f"Results/{db_name}/Plots/{y_val}.tex")
    # set the title
    # two columns for the legend
    plt.legend(ncol=2)
    plt.title(f"{db_name}, {y_val}")
    # set y-axis from 0 to 100
    plt.ylim(0, 100)
    plt.savefig(f"Results/{db_name}/Plots/{db_name}_{y_val}.png")
    plt.show()


def evaluateGraphLearningNN(db_name, ids, path='Results/'):
    evaluation = {}
    for id in ids:
        id_str = str(id).zfill(6)
        # load the data from Results/{db_name}/Results/{db_name}_{id_str}_Results_run_id_{run_id}.csv as a pandas dataframe for all run_ids in the directory
        # ge all those files
        files = []
        network_files = []
        for file in os.listdir(f"Results/{db_name}/Results"):
            if file.startswith(f"{db_name}_{id_str}_Results_run_id_") and file.endswith(".csv"):
                files.append(file)
            elif file.startswith(f"{db_name}_{id_str}_Network") and file.endswith(".txt"):
                network_files.append(file)

        df_all = None
        for i, file in enumerate(files):
            df = pd.read_csv(f"Results/{db_name}/Results/{file}", delimiter=";")
            # concatenate the dataframes
            if df_all is None:
                df_all = df
            else:
                df_all = pd.concat([df_all, df], ignore_index=True)

        # create a new column RunNumberValidationNumber that is the concatenation of RunNumber and ValidationNumber
        df_all['RunNumberValidationNumber'] = df_all['RunNumber'].astype(str) + df_all['ValidationNumber'].astype(str)

        # group the data by RunNumberValidationNumber
        groups = df_all.groupby('RunNumberValidationNumber')

        run_groups = df_all.groupby('RunNumber')
        # plot each run
        # for name, group in run_groups:
        #    group['TestAccuracy'].plot()
        # plt.show()

        indices = []
        # iterate over the groups
        for name, group in groups:
            # get the maximum validation accuracy
            max_val_acc = group['ValidationAccuracy'].max()
            # get the row with the maximum validation accuracy
            max_row = group[group['ValidationAccuracy'] == max_val_acc]
            # get the minimum validation loss if column exists
            #if 'ValidationLoss' in group.columns:
            #    max_val_acc = group['ValidationLoss'].min()
            #    max_row = group[group['ValidationLoss'] == max_val_acc]

            # get row with the minimum validation loss
            min_val_loss = max_row['ValidationLoss'].min()
            max_row = group[group['ValidationLoss'] == min_val_loss]
            max_row = max_row.iloc[-1]
            # get the index of the row
            index = max_row.name
            indices.append(index)

        # get the rows with the indices
        df_validation = df_all.loc[indices]
        mean_validation = df_validation.mean(numeric_only=True)
        std_validation = df_validation.std(numeric_only=True)
        # print epoch accuracy
        print(
            f"Id: {id} Average Epoch Accuracy: {mean_validation['EpochAccuracy']} +/- {std_validation['EpochAccuracy']}")
        print(
            f"Id: {id} Average Validation Accuracy: {mean_validation['ValidationAccuracy']} +/- {std_validation['ValidationAccuracy']}")
        # if name is NCI1, then group by the ValidationNumber
        if db_name == 'NCI1' or db_name == 'ENZYMES' or db_name == 'PROTEINS' or db_name == 'DD' or db_name == 'IMDB-BINARY' or db_name == 'IMDB-MULTI' or db_name == "SYNTHETICnew" or db_name == "DHFR" or db_name == "NCI109" or db_name == "Mutagenicity" or db_name == "MUTAG":
            df_validation = df_validation.groupby('ValidationNumber').mean(numeric_only=True)
        else:
            df_validation = df_validation.groupby('RunNumber').mean(numeric_only=True)
        # get the average and deviation over all runs
        epoch_loss_column_name = None
        for col in df_all.columns:
            if 'EpochLoss' in col:
                epoch_loss_column_name = col
                break
        if epoch_loss_column_name is None:
            print("No column found that contains 'EpochLoss'")
            return
        df_validation[epoch_loss_column_name] *= df_validation['TrainingSize']
        df_validation['TestAccuracy'] *= df_validation['TestSize']
        df_validation['ValidationAccuracy'] *= df_validation['ValidationSize']
        df_validation['ValidationLoss'] *= df_validation['ValidationSize']
        avg = df_validation.mean(numeric_only=True)

        avg[epoch_loss_column_name] /= avg['TrainingSize']
        avg['TestAccuracy'] /= avg['TestSize']
        avg['ValidationAccuracy'] /= avg['ValidationSize']
        avg['ValidationLoss'] /= avg['ValidationSize']

        std = df_validation.std(numeric_only=True)
        std[epoch_loss_column_name] /= avg['TrainingSize']
        std['TestAccuracy'] /= avg['TestSize']
        std['ValidationAccuracy'] /= avg['ValidationSize']
        std['ValidationLoss'] /= avg['ValidationSize']

        # print the avg and std achieved by the highest validation accuracy
        print(f"Id: {id} Average Test Accuracy: {avg['TestAccuracy']} +/- {std['TestAccuracy']}")

        # open network file and read the network
        network_legend = {}
        with open(f"Results/{db_name}/Results/{network_files[0]}", "r") as f:
            # get first line
            line = f.readline()
            # get string between [ and ]
            line = line[line.find("[") + 1:line.find("]")]
            # split by , not in ''
            line = line.split(", ")
            # join the strings with ;
            line = ";".join(line)
            id = file.split('_')[1]
            network_legend[id] = f'Id:{id}, {line}'
        # check if ValidationLoss exists
        if 'ValidationLoss' in df_all.columns:
            evaluation[id] = [avg['TestAccuracy'], std['TestAccuracy'], mean_validation['ValidationAccuracy'],
                              std_validation['ValidationAccuracy'], network_legend[id],
                              mean_validation['ValidationLoss'], std_validation['ValidationLoss']]
        else:
            evaluation[id] = [avg['TestAccuracy'], std['TestAccuracy'], mean_validation['ValidationAccuracy'],
                              std_validation['ValidationAccuracy'], network_legend[id]]

    # print all evaluation items start with id and network then validation and test accuracy
    # round all floats to 2 decimal places
    for key, value in evaluation.items():
        value[0] = round(value[0], 2)
        value[1] = round(value[1], 2)
        value[2] = round(value[2], 2)
        value[3] = round(value[3], 2)
        print(f"{value[4]} Validation Accuracy: {value[2]} +/- {value[3]} Test Accuracy: {value[0]} +/- {value[1]}")

    # print the evaluation items with the k highest validation accuracies
    print(f"Top 5 Validation Accuracies for {db_name}")
    k = 5
    sorted_evaluation = sorted(evaluation.items(), key=lambda x: x[1][2], reverse=True)

    for i in range(min(k, len(sorted_evaluation))):
        if len(sorted_evaluation[i][1]) > 5:
            sorted_evaluation = sorted(sorted_evaluation, key=lambda x: x[1][5], reverse=False)
            print(
                f"{sorted_evaluation[i][1][4]} Validation Loss: {sorted_evaluation[i][1][5]} +/- {sorted_evaluation[i][1][6]} Validation Accuracy: {sorted_evaluation[i][1][2]} +/- {sorted_evaluation[i][1][3]} Test Accuracy: {sorted_evaluation[i][1][0]} +/- {sorted_evaluation[i][1][1]}")
        else:
            print(
                f"{sorted_evaluation[i][1][4]} Validation Accuracy: {sorted_evaluation[i][1][2]} +/- {sorted_evaluation[i][1][3]} Test Accuracy: {sorted_evaluation[i][1][0]} +/- {sorted_evaluation[i][1][1]}")


def model_selection_evaluation(db_name, evaluate_best_model=False, evaluate_validation_only=False, experiment_config=None, get_best_model=False, print_results=False) -> int:
    '''
    Evaluate the model selection results for a specific database
    :param db_name: the name of the database
    :param print_results: print the results to the console
    :param evaluate_best_model: evaluate the best model
    :param get_best_model: get the best model
    :param evaluate_validation_only: evaluate only the validation set (get the mean best epoch accuracy and the std)
    :return: the configuration id of the model with the overall best validation accuracy (+ minimum validation loss)

    '''
    result_path = experiment_config['paths']['results']
    evaluation_type = experiment_config.get('evaluation_type', 'accuracy')
    best_configuration_id = None
    if get_best_model:
        # find the best configuration id
        best_configuration_id = 0
        data = None
        # load the data
        if evaluate_best_model:
            data = pd.read_csv(result_path.joinpath(db_name).joinpath('summary_best.csv'))
        else:
            data = pd.read_csv(result_path.joinpath(db_name).joinpath('summary.csv'))
        if evaluation_type == 'accuracy':
            # get rows with the maximum validation accuracy
            best_configuration_ids = data[data['Validation Accuracy Mean'] == data['Validation Accuracy Mean'].max()]
            # if there are multiple rows with the maximum validation accuracy, get the one with the minimum validation loss
            if best_configuration_ids.shape[0] > 1:
                best_configuration_id = best_configuration_ids[best_configuration_ids['Validation Loss Mean'] == best_configuration_ids['Validation Loss Mean'].min()]['ConfigurationId'].values[0]
            else:
                best_configuration_id = best_configuration_ids['ConfigurationId'].values[0]
        elif evaluation_type == 'loss':
            # get the rows with the minimum validation loss
            best_configuration_ids = data[data['Validation Loss Mean'] == data['Validation Loss Mean'].min()]
            # if there are multiple rows with the minimum validation loss, get the one with the maximum validation accuracy
            if best_configuration_ids.shape[0] > 1:
                best_configuration_id = best_configuration_ids[best_configuration_ids['Validation Accuracy Mean'] == best_configuration_ids['Validation Accuracy Mean'].max()]['ConfigurationId'].values[0]
            else:
                best_configuration_id = best_configuration_ids['ConfigurationId'].values[0]
    else:
        # add absolute path to path
        result_path = Path(os.path.abspath(result_path))
        if print_results:
            print(f"Model Selection Evaluation for {db_name}")
        db = None
        # get all run ids from search path
        search_path = result_path.joinpath(db_name).joinpath('Results')
        # check if path exists
        if not search_path.exists():
            print(f"Path {search_path} does not exist")
            return 0
        for file in os.listdir(search_path):
            if file.find('run_id') != -1 and file.find('validation_step') != -1 and file.endswith(".csv"):
                df_local = pd.read_csv(search_path.joinpath(file), delimiter=";")
                df_local['ConfigurationId'] = int(file.split('Configuration_')[-1].split('_')[0])
                if not evaluate_best_model:
                    if file.find('Best_Configuration') == -1:
                        if db is None:
                            db = df_local
                        else:
                            db = pd.concat([db, df_local], ignore_index=True)
                else:
                    if file.find('Best_Configuration') != -1:
                        if db is None:
                            db = df_local
                        else:
                            db = pd.concat([db, df_local], ignore_index=True)
        # group by ConfigurationId and RunNumber
        groups_db = None
        if evaluate_validation_only:
            with open(result_path.joinpath(db_name).joinpath('summary_sota.csv'), 'w') as f:
                f.write(
                    'ConfigurationId,RunId,Epoch,Epoch Accuracy Mean,Epoch Accuracy Std,Epoch Loss Mean,Epoch Loss Std,Validation Accuracy Mean,Validation Accuracy Std,Validation Loss Mean,Validation Loss Std\n')
            groups_db = db.groupby(['ConfigurationId', 'RunNumber'])
            for name, group in groups_db:
                # merge all rows with the same Epoch and get the mean resp. std (do not remove Epoch column)
                mean_group = group.groupby('Epoch').mean(numeric_only=True).reset_index()
                std_group = group.groupby('Epoch').std(numeric_only=True).reset_index()
                if evaluation_type == 'accuracy':
                    # get the maximum validation accuracy
                    max_val_acc = mean_group['ValidationAccuracy'].max()
                    # get the row with the maximum validation accuracy
                    max_row = mean_group[mean_group['ValidationAccuracy'] == max_val_acc]

                elif evaluation_type == 'loss':
                    # get the minimum validation loss if column exists
                    if 'ValidationLoss' in mean_group.columns:
                        max_val_acc = mean_group['ValidationLoss'].min()
                        max_row = mean_group[mean_group['ValidationLoss'] == max_val_acc]
                else:
                    raise ValueError(f"evaluation_type {evaluation_type} not supported. Please use 'accuracy' or 'loss'")

                # get row with the minimum validation loss
                min_val_loss = max_row['ValidationLoss'].min()
                max_row = max_row[max_row['ValidationLoss'] == min_val_loss]
                # get the maximum epoch of the series max_row
                max_epoch = max_row['Epoch'].max()
                max_mean = mean_group[mean_group['Epoch'] == max_epoch].iloc[-1]
                max_std = std_group[std_group['Epoch'] == max_epoch].iloc[-1]

                epoch_loss_column_name = None
                for col in db.columns:
                    if 'EpochLoss' in col:
                        epoch_loss_column_name = col
                        break
                if epoch_loss_column_name is None:
                    print("No column found that contains 'EpochLoss'")
                    return

                # write the results to summary_sota.csv using the
                with open(result_path.joinpath(db_name).joinpath('summary_sota.csv'), 'a') as f:
                    f.write(f"{int(max_mean['ConfigurationId'])},{int(max_mean['RunNumber'])},{int(max_mean['Epoch'])},{max_mean['EpochAccuracy']},{max_std['EpochAccuracy']},{max_mean[epoch_loss_column_name]},{max_std[epoch_loss_column_name]},{max_mean['ValidationAccuracy']},{max_std['ValidationAccuracy']},{max_mean['ValidationLoss']},{max_std['ValidationLoss']}\n")

        else:
            if db is not None:
                groups_db = db.groupby(['ConfigurationId', 'RunNumber', 'ValidationNumber'])
            else:
                if evaluate_best_model:
                    print(f"No files found for {db_name} with Best_Configuration")
                else:
                    print(f"No files found for {db_name}")
                    return



            indices = []
            # iterate over the groups
            for name, group in groups_db:
                if is_pruning(experiment_config):
                    group = group[group['Epoch'] >= group['Epoch'].max() - experiment_config['pruning']['pruning_step']]
                if evaluation_type == 'accuracy':
                    # get the maximum validation accuracy
                    max_val_acc = group['ValidationAccuracy'].max()
                    # get the row with the maximum validation accuracy
                    max_row = group[group['ValidationAccuracy'] == max_val_acc]
                elif evaluation_type == 'loss':
                    # get the minimum validation loss if column exists
                    if 'ValidationLoss' in group.columns:
                        max_val_acc = group['ValidationLoss'].min()
                        max_row = group[group['ValidationLoss'] == max_val_acc]

                # get row with the minimum validation loss
                min_val_loss = max_row['ValidationLoss'].min()
                max_row = group[group['ValidationLoss'] == min_val_loss]
                max_row = max_row.iloc[-1]
                # get the index of the row
                index = max_row.name
                indices.append(index)

            # get the rows with the indices
            df_validation = db.loc[indices]
            # split into groups by ConfigurationId and RunNumber
            validation_groups = df_validation.groupby(['ConfigurationId', 'RunNumber'])
            # write headers to file
            if evaluate_best_model:
                with open(result_path.joinpath(db_name).joinpath('summary_best.csv'), 'w') as f:
                    f.write('Seed,ConfigurationId,RunId,Epoch Mean,Epoch Std,Epoch Accuracy Mean,Epoch Accuracy Std,Epoch Loss Mean,Epoch Loss Std,Validation Accuracy Mean,Validation Accuracy Std,Validation Loss Mean,Validation Loss Std,Test Accuracy Mean,Test Accuracy Std,Test Loss Mean,Test Loss Std\n')
            else:
                with open(result_path.joinpath(db_name).joinpath('summary.csv'), 'w') as f:
                    f.write('Seed,ConfigurationId,RunId,Epoch Mean,Epoch Std,Epoch Accuracy Mean,Epoch Accuracy Std,Epoch Loss Mean,Epoch Loss Std,Validation Accuracy Mean,Validation Accuracy Std,Validation Loss Mean,Validation Loss Std,Test Accuracy Mean,Test Accuracy Std,Test Loss Mean,Test Loss Std\n')

            # find column name that contains EpochLoss
            epoch_loss_column_name = None
            for col in df_validation.columns:
                if 'EpochLoss' in col:
                    epoch_loss_column_name = col
                    break
            if epoch_loss_column_name is None:
                print("No column found that contains 'EpochLoss'")
                return


            for name, group in validation_groups:
                group[epoch_loss_column_name] *= group['TrainingSize']
                group['Epoch'] *= group['TrainingSize']
                group['EpochAccuracy'] *= group['TrainingSize']
                group['TestAccuracy'] *= group['TestSize']
                group['TestLoss'] *= group['TestSize']
                group['ValidationAccuracy'] *= group['ValidationSize']
                group['ValidationLoss'] *= group['ValidationSize']
                avg = group.mean(numeric_only=True)

                avg[epoch_loss_column_name] /= avg['TrainingSize']
                avg['Epoch'] /= avg['TrainingSize']
                avg['EpochAccuracy'] /= avg['TrainingSize']
                avg['TestAccuracy'] /= avg['TestSize']
                avg['TestLoss'] /= avg['TestSize']
                avg['ValidationAccuracy'] /= avg['ValidationSize']
                avg['ValidationLoss'] /= avg['ValidationSize']

                std = group.std(numeric_only=True)
                std[epoch_loss_column_name] /= avg['TrainingSize']
                std['Epoch'] /= avg['TrainingSize']
                std['EpochAccuracy'] /= avg['TrainingSize']
                std['TestAccuracy'] /= avg['TestSize']
                std['TestLoss'] /= avg['TestSize']
                std['ValidationAccuracy'] /= avg['ValidationSize']
                std['ValidationLoss'] /= avg['ValidationSize']

                configuration_id = int(avg['ConfigurationId'])
                run_id = int(avg['RunNumber'])
                # write to file
                seed = int(avg['Seed'])
                if evaluate_best_model:
                    with open(result_path.joinpath(db_name).joinpath('summary_best.csv'), 'a') as f:
                        f.write(f"{seed},{configuration_id},{run_id},{avg['Epoch']},{std['Epoch']},{avg['EpochAccuracy']},{std['EpochAccuracy']},{avg[epoch_loss_column_name]},{std[epoch_loss_column_name]},{avg['ValidationAccuracy']},{std['ValidationAccuracy']},{avg['ValidationLoss']},{std['ValidationLoss']},{avg['TestAccuracy']},{std['TestAccuracy']},{avg['TestLoss']},{std['TestLoss']}\n")
                else:
                    with open(result_path.joinpath(db_name).joinpath('summary.csv'), 'a') as f:
                        f.write(f"{seed},{configuration_id},{run_id},{avg['Epoch']},{std['Epoch']},{avg['EpochAccuracy']},{std['EpochAccuracy']},{avg[epoch_loss_column_name]},{std[epoch_loss_column_name]},{avg['ValidationAccuracy']},{std['ValidationAccuracy']},{avg['ValidationLoss']},{std['ValidationLoss']},{avg['TestAccuracy']},{std['TestAccuracy']},{avg['TestLoss']},{std['TestLoss']}\n")


            if evaluate_best_model:
                # write summary_best_mean.csv
                # load summary best model
                summary_best_model = pd.read_csv(result_path.joinpath(db_name).joinpath('summary_best.csv'))
                # remove
                # average over all rows
                summary_best_model_mean = summary_best_model.mean(numeric_only=True)
                # drop column RunId
                summary_best_model_mean = summary_best_model_mean.drop('RunId')
                config_id = int(summary_best_model_mean['ConfigurationId'])
                # write to file
                with open(result_path.joinpath(db_name).joinpath('summary_best_mean.csv'), 'w') as f:
                    f.write('ConfigurationId,Epoch Mean,Epoch Std,Epoch Accuracy Mean,Epoch Accuracy Std,Epoch Loss Mean,Epoch Loss Std,Validation Accuracy Mean,Validation Accuracy Std,Validation Loss Mean,Validation Loss Std,Test Accuracy Mean,Test Accuracy Std,Test Loss Mean,Test Loss Std\n')
                    f.write(f'{config_id},{summary_best_model_mean["Epoch Mean"]},{summary_best_model_mean["Epoch Std"]},{summary_best_model_mean["Epoch Accuracy Mean"]},{summary_best_model_mean["Epoch Accuracy Std"]},{summary_best_model_mean["Epoch Loss Mean"]},{summary_best_model_mean["Epoch Loss Std"]},{summary_best_model_mean["Validation Accuracy Mean"]},{summary_best_model_mean["Validation Accuracy Std"]},{summary_best_model_mean["Validation Loss Mean"]},{summary_best_model_mean["Validation Loss Std"]},{summary_best_model_mean["Test Accuracy Mean"]},{summary_best_model_mean["Test Accuracy Std"]},{summary_best_model_mean["Test Loss Mean"]},{summary_best_model_mean["Test Loss Std"]}')

    return best_configuration_id


def model_selection_evaluation_mae(db_name, path:Path, ids=None):
    # add absolute path to path
    path = os.path.abspath(path)
    print(f"Model Selection Evaluation for {db_name}")
    evaluation = {}

    if ids is None:
        # get all ids
        ids = []
        search_path = path.joinpath(db_name).joinpath('Results')
        for file in os.listdir(search_path):
            if file.endswith(".txt"):
                id = int(file.split('_')[-2])
                ids.append(id)
        # sort the ids
        ids.sort()

    for id in ids:
        id_str = str(id).zfill(6)
        # load the data from Results/{db_name}/Results/{db_name}_{id_str}_Results_run_id_{run_id}.csv as a pandas dataframe for all run_ids in the directory
        # ge all those files
        files = []
        network_files = []
        search_path = path.joinpath(db_name).joinpath('Results')
        for file in os.listdir(search_path):
            if file.startswith(f"{db_name}_Configuration_{id_str}_Results_run_id_") and file.endswith(".csv"):
                files.append(file)
            elif file.startswith(f"{db_name}_Configuration_{id_str}_Network") and file.endswith(".txt"):
                network_files.append(file)

        # check that there are 10 files, for CSL 5 and for ZINC 1
        if len(files) != 10 and (db_name == 'ZINC' and len(files) != 1):
            print(f"Id: {id} has {len(files)} files")
            continue
        df_all = None
        for i, file in enumerate(files):
            file_path = path.joinpath(db_name).joinpath('Results').joinpath(file)
            df = pd.read_csv(file_path, delimiter=";")
            # concatenate the dataframes
            if df_all is None:
                df_all = df
            else:
                df_all = pd.concat([df_all, df], ignore_index=True)

        # create a new column RunNumberValidationNumber that is the concatenation of RunNumber and ValidationNumber
        df_all['RunNumberValidationNumber'] = df_all['RunNumber'].astype(str) + df_all['ValidationNumber'].astype(str)

        # check if df_all has a row
        if df_all.shape[0] != 0:

            # group the data by RunNumberValidationNumber
            groups = df_all.groupby('RunNumberValidationNumber')

            indices = []
            # iterate over the groups
            for name, group in groups:

                max_val_acc = group['ValidationLoss'].min()
                max_row = group[group['ValidationLoss'] == max_val_acc]

                # get row with the minimum validation loss
                min_val_loss = max_row['ValidationLoss'].min()
                max_row = group[group['ValidationLoss'] == min_val_loss]
                max_row = max_row.iloc[-1]
                # get the index of the row
                index = max_row.name
                indices.append(index)

            # get the rows with the indices
            df_validation = df_all.loc[indices]
            df_validation = df_validation.groupby('ValidationNumber').mean(numeric_only=True)

            # get the average and deviation over all runs
            epoch_loss_column_name = None
            for col in df_validation.columns:
                if 'EpochLoss' in col:
                    epoch_loss_column_name = col
                    break
            if epoch_loss_column_name is None:
                print("No column found that contains 'EpochLoss'")
                return
            df_validation[epoch_loss_column_name] *= df_validation['TrainingSize']
            df_validation['Epoch'] *= df_validation['TrainingSize']
            df_validation['EpochAccuracy'] *= df_validation['TrainingSize']
            df_validation['TestAccuracy'] *= df_validation['TestSize']
            df_validation['TestLoss'] *= df_validation['TestSize']
            df_validation['ValidationAccuracy'] *= df_validation['ValidationSize']
            df_validation['ValidationLoss'] *= df_validation['ValidationSize']
            avg = df_validation.mean(numeric_only=True)

            avg[epoch_loss_column_name] /= avg['TrainingSize']
            avg['Epoch'] /= avg['TrainingSize']
            avg['EpochAccuracy'] /= avg['TrainingSize']
            avg['TestAccuracy'] /= avg['TestSize']
            avg['TestLoss'] /= avg['TestSize']
            avg['ValidationAccuracy'] /= avg['ValidationSize']
            avg['ValidationLoss'] /= avg['ValidationSize']

            std = df_validation.std(numeric_only=True)
            std[epoch_loss_column_name] /= avg['TrainingSize']
            std['Epoch'] /= avg['TrainingSize']
            std['EpochAccuracy'] /= avg['TrainingSize']
            std['TestAccuracy'] /= avg['TestSize']
            std['TestLoss'] /= avg['TestSize']
            std['ValidationAccuracy'] /= avg['ValidationSize']
            std['ValidationLoss'] /= avg['ValidationSize']

            # print the avg and std achieved by the highest validation loss
            print(f"Id: {id} "
                  f"Epoch: {avg['Epoch']} +/- {std['Epoch']} "
                  f"Average Training Loss: {avg[epoch_loss_column_name]} +/- {std[epoch_loss_column_name]} "
                  f"Average Validation Loss: {avg['ValidationLoss']} +/- {std['ValidationLoss']} "
                  f"Average Test Loss: {avg['TestLoss']} +/- {std['TestLoss']} ")


            evaluation[id] = [avg[epoch_loss_column_name], std[epoch_loss_column_name],avg['ValidationLoss'], std['ValidationLoss'],avg['TestLoss'], std['TestLoss']]


    # print all evaluation items start with id and network then validation and test accuracy
    # round all floats to 2 decimal places
    for key, value in evaluation.items():
        value[0] = round(value[0], 6)
        value[1] = round(value[1], 6)
        value[2] = round(value[2], 6)
        value[3] = round(value[3], 6)
        value[4] = round(value[4], 6)
        value[5] = round(value[5], 6)

    # print the evaluation items with the k highest validation accuracies
    k = 10
    print(f"Top {k} Validation Accuracies for {db_name}")


    sort_key = 2

    sorted_evaluation = sorted(evaluation.items(), key=lambda x: x[1][sort_key], reverse=False)

    # write results to file called summary.csv
    summary_path = path.joinpath(db_name).joinpath('summary.csv')
    with open(summary_path, 'w') as f:
        f.write('Id, Epoch Loss, Epoch Loss Std, Validation Loss, Validation Loss Std, Test Loss, Test Loss Std\n')
        for i in range(min(k, len(sorted_evaluation))):
            sorted_evaluation = sorted(sorted_evaluation, key=lambda x: x[1][sort_key], reverse=False)
            f.write(f"{sorted_evaluation[i][0]}, {sorted_evaluation[i][1][0]}, {sorted_evaluation[i][1][1]}, {sorted_evaluation[i][1][2]}, {sorted_evaluation[i][1][3]}, {sorted_evaluation[i][1][4]}, {sorted_evaluation[i][1][5]}\n")

    for i in range(min(k, len(sorted_evaluation))):
        sorted_evaluation = sorted(sorted_evaluation, key=lambda x: x[1][sort_key], reverse=False)
        print(
            f" Id: {sorted_evaluation[i][0]} "
            f" Epoch Loss: {sorted_evaluation[i][1][0]} +/- {sorted_evaluation[i][1][1]} "
            f" Validation Loss: {sorted_evaluation[i][1][2]} +/- {sorted_evaluation[i][1][3]} "
            f"Test Loss: {sorted_evaluation[i][1][4]} +/- {sorted_evaluation[i][1][5]}")