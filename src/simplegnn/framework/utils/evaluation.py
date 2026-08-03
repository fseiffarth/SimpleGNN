import os
from pathlib import Path

import pandas as pd

from simplegnn.utils.utils import is_pruning


def model_selection_evaluation(db_name, evaluate_best_model=False, evaluate_validation_only=False, experiment_config=None, get_best_model=False, print_results=False) -> int:
    """
    Perform comprehensive model selection evaluation and generate summary statistics.

    This function orchestrates the complete evaluation pipeline:
    1. Loads all result CSV files for a dataset
    2. Performs model selection (best epoch per configuration based on validation metric)
    3. Aggregates results across validation folds
    4. Generates summary CSV files with mean ± std for all metrics
    5. Optionally identifies the single best configuration

    Parameters
    ----------
    db_name : str
        Dataset name. Results are expected at <results_path>/<db_name>/Results/.
    evaluate_best_model : bool, optional
        If True, evaluate results from the best configuration's re-runs
        (expects files with 'Best_Configuration' in the name). Default: False.
    evaluate_validation_only : bool, optional
        If True, only compute mean validation metrics without full evaluation.
        Default: False.
    experiment_config : dict, optional
        Experiment configuration dictionary containing:
        - 'paths']['results']: Path to results directory
        - 'evaluation_type']: 'accuracy' or 'loss' (default: 'accuracy')
    get_best_model : bool, optional
        If True, read pre-existing summary CSV to identify the best configuration ID.
        Default: False.
    print_results : bool, optional
        If True, print evaluation results to console. Default: False.

    Returns
    -------
    int
        Configuration ID of the best model. Selection criteria:
        - If evaluation_type='accuracy': Highest validation accuracy (ties broken by lowest loss)
        - If evaluation_type='loss': Lowest validation loss (ties broken by highest accuracy)
        Returns 0 if no valid results found.

    Notes
    -----
    **Model Selection Logic (per configuration):**
    For each (run_id, validation_id) pair:
    1. Find epoch with best validation metric (max accuracy or min loss)
    2. Extract all metrics at that epoch
    3. Aggregate across folds by grouping on (run_id, validation_id)

    **Aggregation Strategy:**
    Metrics are weighted by dataset size before computing mean/std:
    - Accuracy × Size, then divide by mean size
    - Loss × Size, then divide by mean size

    **Output Files Created:**
    - summary.csv (or summary_best.csv if evaluate_best_model=True):
        Columns: ConfigurationId, Test Accuracy Mean, Test Accuracy Std,
                 Validation Accuracy Mean, Validation Accuracy Std,
                 Validation Loss Mean, Validation Loss Std, etc.

    **Pruning Detection:**
    The function uses is_pruning() utility to detect if neural architecture search
    with pruning was performed, which affects result aggregation logic.

    **get_best_model Mode:**
    Instead of processing CSVs, reads the summary file and selects the row with:
    - Maximum 'Validation Accuracy Mean' (if evaluation_type='accuracy'),
    - Maximum 'Validation AUC Mean' (if evaluation_type='roc_auc'), or
    - Minimum 'Validation Loss Mean' (if evaluation_type='loss')

    ``evaluation_type='roc_auc'`` requires the run to have been configured with
    ``evaluation_metric: roc_auc`` (see model_configuration.py), which is what
    writes the EpochAUC/ValidationAUC/TestAUC columns the selection reads. The
    AUC columns are appended after the fixed accuracy/loss columns in the
    summary files, so accuracy-only experiments keep their previous layout.

    Raises
    ------
    FileNotFoundError
        If results directory doesn't exist or expected CSV files are missing.
    """
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
        elif evaluation_type == 'roc_auc':
            # get rows with the maximum validation AUC (written only when the run
            # used `evaluation_metric: roc_auc`, see model_configuration.py)
            if 'Validation AUC Mean' not in data.columns:
                raise ValueError(
                    "evaluation_type 'roc_auc' requires AUC columns in the summary file. "
                    "Set `evaluation_metric: roc_auc` in the hyperparameter config and rerun the experiment.")
            best_configuration_ids = data[data['Validation AUC Mean'] == data['Validation AUC Mean'].max()]
            # tie-break on the minimum validation loss, as in the accuracy branch
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

                elif evaluation_type == 'roc_auc':
                    # get the row with the maximum validation AUC
                    max_val_auc = mean_group['ValidationAUC'].max()
                    max_row = mean_group[mean_group['ValidationAUC'] == max_val_auc]

                elif evaluation_type == 'loss':
                    # get the minimum validation loss if column exists
                    if 'ValidationLoss' in mean_group.columns:
                        max_val_acc = mean_group['ValidationLoss'].min()
                        max_row = mean_group[mean_group['ValidationLoss'] == max_val_acc]
                else:
                    raise ValueError(f"evaluation_type {evaluation_type} not supported. Please use 'accuracy', 'roc_auc' or 'loss'")

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
                elif evaluation_type == 'roc_auc':
                    # get the row with the maximum validation AUC
                    max_val_auc = group['ValidationAUC'].max()
                    max_row = group[group['ValidationAUC'] == max_val_auc]
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
            # AUC columns exist only for runs configured with `evaluation_metric:
            # roc_auc`; they are appended after the fixed columns so the existing
            # accuracy-only schema keeps its exact layout.
            auc_columns = ['EpochAUC', 'ValidationAUC', 'TestAUC']
            has_auc = all(col in df_validation.columns for col in auc_columns)
            summary_header = ('Seed,ConfigurationId,RunId,Epoch Mean,Epoch Std,Epoch Accuracy Mean,Epoch Accuracy Std,'
                              'Epoch Loss Mean,Epoch Loss Std,Validation Accuracy Mean,Validation Accuracy Std,'
                              'Validation Loss Mean,Validation Loss Std,Test Accuracy Mean,Test Accuracy Std,'
                              'Test Loss Mean,Test Loss Std')
            if has_auc:
                summary_header += (',Epoch AUC Mean,Epoch AUC Std,Validation AUC Mean,Validation AUC Std,'
                                   'Test AUC Mean,Test AUC Std')
            summary_header += '\n'

            # write headers to file
            if evaluate_best_model:
                with open(result_path.joinpath(db_name).joinpath('summary_best.csv'), 'w') as f:
                    f.write(summary_header)
            else:
                with open(result_path.joinpath(db_name).joinpath('summary.csv'), 'w') as f:
                    f.write(summary_header)

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
                if has_auc:
                    # size-weighted, exactly like the accuracy columns above
                    group['EpochAUC'] *= group['TrainingSize']
                    group['ValidationAUC'] *= group['ValidationSize']
                    group['TestAUC'] *= group['TestSize']
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

                auc_fields = ''
                if has_auc:
                    avg['EpochAUC'] /= avg['TrainingSize']
                    avg['ValidationAUC'] /= avg['ValidationSize']
                    avg['TestAUC'] /= avg['TestSize']
                    std['EpochAUC'] /= avg['TrainingSize']
                    std['ValidationAUC'] /= avg['ValidationSize']
                    std['TestAUC'] /= avg['TestSize']
                    auc_fields = (f",{avg['EpochAUC']},{std['EpochAUC']},"
                                  f"{avg['ValidationAUC']},{std['ValidationAUC']},"
                                  f"{avg['TestAUC']},{std['TestAUC']}")

                configuration_id = int(avg['ConfigurationId'])
                run_id = int(avg['RunNumber'])
                # write to file
                seed = int(avg['Seed'])
                summary_row = (f"{seed},{configuration_id},{run_id},{avg['Epoch']},{std['Epoch']},"
                               f"{avg['EpochAccuracy']},{std['EpochAccuracy']},"
                               f"{avg[epoch_loss_column_name]},{std[epoch_loss_column_name]},"
                               f"{avg['ValidationAccuracy']},{std['ValidationAccuracy']},"
                               f"{avg['ValidationLoss']},{std['ValidationLoss']},"
                               f"{avg['TestAccuracy']},{std['TestAccuracy']},"
                               f"{avg['TestLoss']},{std['TestLoss']}{auc_fields}\n")
                if evaluate_best_model:
                    with open(result_path.joinpath(db_name).joinpath('summary_best.csv'), 'a') as f:
                        f.write(summary_row)
                else:
                    with open(result_path.joinpath(db_name).joinpath('summary.csv'), 'a') as f:
                        f.write(summary_row)


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
                mean_header = ('ConfigurationId,Epoch Mean,Epoch Std,Epoch Accuracy Mean,Epoch Accuracy Std,'
                               'Epoch Loss Mean,Epoch Loss Std,Validation Accuracy Mean,Validation Accuracy Std,'
                               'Validation Loss Mean,Validation Loss Std,Test Accuracy Mean,Test Accuracy Std,'
                               'Test Loss Mean,Test Loss Std')
                mean_row = (f'{config_id},{summary_best_model_mean["Epoch Mean"]},{summary_best_model_mean["Epoch Std"]},'
                            f'{summary_best_model_mean["Epoch Accuracy Mean"]},{summary_best_model_mean["Epoch Accuracy Std"]},'
                            f'{summary_best_model_mean["Epoch Loss Mean"]},{summary_best_model_mean["Epoch Loss Std"]},'
                            f'{summary_best_model_mean["Validation Accuracy Mean"]},{summary_best_model_mean["Validation Accuracy Std"]},'
                            f'{summary_best_model_mean["Validation Loss Mean"]},{summary_best_model_mean["Validation Loss Std"]},'
                            f'{summary_best_model_mean["Test Accuracy Mean"]},{summary_best_model_mean["Test Accuracy Std"]},'
                            f'{summary_best_model_mean["Test Loss Mean"]},{summary_best_model_mean["Test Loss Std"]}')
                if has_auc:
                    mean_header += (',Epoch AUC Mean,Epoch AUC Std,Validation AUC Mean,Validation AUC Std,'
                                    'Test AUC Mean,Test AUC Std')
                    mean_row += (f',{summary_best_model_mean["Epoch AUC Mean"]},{summary_best_model_mean["Epoch AUC Std"]},'
                                 f'{summary_best_model_mean["Validation AUC Mean"]},{summary_best_model_mean["Validation AUC Std"]},'
                                 f'{summary_best_model_mean["Test AUC Mean"]},{summary_best_model_mean["Test AUC Std"]}')
                # write to file
                with open(result_path.joinpath(db_name).joinpath('summary_best_mean.csv'), 'w') as f:
                    f.write(mean_header + '\n')
                    f.write(mean_row)

    return best_configuration_id
