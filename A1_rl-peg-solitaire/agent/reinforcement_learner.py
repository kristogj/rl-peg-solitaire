from actor import Actor
from utils import get_critic
import logging


class ReinforcementLearner:

    def __init__(self, config):
        logging.info("Setting up the ReinforcementLearner")
        self.config = config
        self.actor = Actor(config["Actor"])
        self.critic = get_critic(config["Critic"])

    def train(self):

        for episode in range(1, self.config["episodes"] + 1):

            # TODO: Reset eligibilities in actor and critic
            # TODO: Init start state and start action
            state, action = None, None

            while True:
                pass

        return
