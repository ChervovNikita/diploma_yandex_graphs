"""Bind a refusing CLI's own local config without changing host admission."""


def negative_cli_environment(parent_environment, local_config_pin):
    if not isinstance(local_config_pin.get('path'), str) or not local_config_pin['path'].startswith('/') \
            or not isinstance(local_config_pin.get('sha256'), str) or len(local_config_pin['sha256']) != 64:
        raise ValueError('Exact local worker config receipt required')
    environment = dict(parent_environment)
    environment['BENCHMARK_HOST_CONFIG'] = local_config_pin['path']
    environment['BENCHMARK_HOST_CONFIG_SHA256'] = local_config_pin['sha256']
    return environment
