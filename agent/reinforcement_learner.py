from actor import Actor
from utils import get_critic
import logging
from critic import NeuralCritic, TableCritic


class ReinforcementLearner:

    def __init__(self, player, config):
        logging.info("Setting up the ReinforcementLearner")
        self.config = config
        self.actor = Actor(player, config["Actor"])
        self.critic = get_critic(config["Critic"])

    def get_actor(self):
        """
        Return the Actor
        :return: Actor
        """
        return self.actor

    def get_critic(self):
        """
        Return the Critic
        :return: Critic
        """
        return self.critic

    def update(self, state, action):
        """
        Step 6 of the actor-critic algorithm require update in actor and critic for all SAP in current episode
        :param state: str
        :param action: Action
        :return: None
        """
        # Update Critic
        if self.config["Critic"]["table_lookup"]:
            self.critic.update_value(state)
            self.critic.update_eligibility(state)
        else:
            self.critic.update_weights(state)
            self.critic.update_eligibility()
        # Update Actor
        self.actor.update_policy(state, action)
        self.actor.update_eligibility(state, action)

    def set_eligibility(self, state, action, is_current_state):
        """
        Set the eligibility for both critic and actor to 1 using their update functions
        :param state: str
        :param action: Action
        :param is_current_state: boolean
        :return: None
        """
        if self.config["Critic"]["table_lookup"]:
            self.critic.update_eligibility(state, is_current_state)
        else:
            self.critic.update_eligibility(is_current_state)

        self.actor.update_eligibility(state, action, is_current_state)

    def reset_eligibility(self):
        """
        Reset eligibility for both actor and critic
        :return: None
        """
        self.actor.reset_eligibility()
        self.critic.reset_eligibility()
