from actor import Actor
from utils import get_critic
import logging


class ReinforcementLearner:

    def __init__(self, player, config):
        logging.info("Setting up the ReinforcementLearner")
        self.config = config
        self.actor = Actor(player, config["Actor"])
        self.critic = get_critic(config["Critic"])

    def get_actor(self):
        return self.actor

    def get_critic(self):
        return self.critic
