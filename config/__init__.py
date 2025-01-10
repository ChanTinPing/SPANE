import os
import yaml

class ConfigError(Exception):
    pass

def load_config(env):
    res = None
    for path in config_file_paths(env):
        if os.path.isfile(path):
            with open(path, 'r', encoding='utf-8') as config:
                new_config = yaml.safe_load(config)
            if res is None:
                res = new_config
            else:
                res.update(new_config)
    if res is None:
        raise ConfigError(f"Configuration for env '{env}' not found.")
    return res

def config_file_paths(env):
    """Generate possible config file paths."""
    base_dir = os.path.dirname(__file__)
    paths = [os.path.join(base_dir, 'default.yaml')]
    if env:
        paths.append(os.path.join(base_dir, 'envs', f'{env}.yaml'))
    return paths


class Config:
    def __init__(self, env):
        config = load_config(env)
        self.__dict__.update(config)
        self.env = env
