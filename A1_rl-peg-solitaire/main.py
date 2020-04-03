from utils import load_config
import logging
from environment.sim_world import SimWorld
from visualizer import BoardVisualizer

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler("./app.log", mode="w"),
            logging.StreamHandler()
        ])
    config_path = "configs/task_2.yaml"

    # Load settings for this run
    config = load_config(config_path)
    logging.info(config)

    # Initialize the Simulated World
    SimWorld(config)
