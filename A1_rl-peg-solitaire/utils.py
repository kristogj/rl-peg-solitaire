import yaml


def load_config(path):
    """
    Load the configuration from task_2.yaml.
    """
    return yaml.load(open('configs/task_2.yaml', 'r'), Loader=yaml.SafeLoader)
