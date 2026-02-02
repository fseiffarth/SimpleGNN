def config_paths_to_absolute(experiment_configuration, absolute_path):
    experiment_configuration['paths']['data'] = absolute_path.joinpath(experiment_configuration['paths']['data'])
    experiment_configuration['paths']['results'] = absolute_path.joinpath(experiment_configuration['paths']['results'])
    experiment_configuration['paths']['splits'] = absolute_path.joinpath(experiment_configuration['paths']['splits'])

    experiment_configuration['paths']['labels'] = ''
    if 'labels' in experiment_configuration['paths']:
        experiment_configuration['paths']['labels'] = absolute_path.joinpath(experiment_configuration['paths']['labels'])
    experiment_configuration['paths']['properties'] = ''
    if 'properties' in experiment_configuration['paths']:
        experiment_configuration['paths']['properties'] = absolute_path.joinpath(experiment_configuration['paths']['properties'])


def paths_to_absolute(paths, absolute_path):
    if paths.get('data', None) is not None:
        paths['data'] = absolute_path.joinpath(paths['data'])
    if paths.get('results', None) is not None:
        paths['results'] = absolute_path.joinpath(paths['results'])
    if paths.get('labels', None) is not None:
        paths['labels'] = absolute_path.joinpath(paths['labels'])
    if paths.get('properties', None) is not None:
        paths['properties'] = absolute_path.joinpath(paths['properties'])
    if paths.get('splits', None) is not None:
        paths['splits'] = absolute_path.joinpath(paths['splits'])
