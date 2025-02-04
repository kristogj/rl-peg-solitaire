import yaml
from environment.board import DiamondPegBoard, TrianglePegBoard
from agent.critic import TableCritic, NeuralCritic
import logging


def load_config(path):
    """
    Load the configuration from task_2_table.yaml.
    """
    return yaml.load(open(path, 'r'), Loader=yaml.SafeLoader)


def get_board(config):
    """
    Return the correct board based on what is given in the configurations
    :param config: dict
    :return: PegBoard
    """
    if config["type"] == "d":
        return DiamondPegBoard(config)
    elif config["type"] == "t":
        return TrianglePegBoard(config)


def get_critic(config):
    """
    Return the correct critic based on what is given in the configurations
    :param config: dict
    :return: Critic
    """
    return TableCritic(config) if config["table_lookup"] else NeuralCritic(config)


def init_logger():
    """
    Initialize logger settings
    :return: None
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler("app.log", mode="w"),
            logging.StreamHandler()
        ])


def is_empty(lst):
    """
    Return boolean if the list is empty or not
    :param lst: list
    :return: boolean
    """
    return len(lst) == 0
