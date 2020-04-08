from utils import load_config, init_logger
import logging
from environment.sim_world import SimWorld
from agent.reinforcement_learner import ReinforcementLearner
from plotting import plot_progression_of_learning
import random
import torch
import numpy as np

if __name__ == '__main__':
    init_logger()
    # random.seed(42)
    # torch.manual_seed(42)
    config_path = "configs/task_3_table.yaml"

    # Load settings for this run
    config = load_config(config_path)
    training_config = config["Training"]
    for key in config.keys():
        logging.info("{}: {}".format(key, config[key]))

    # Initialize the Simulated World
    sim_world = SimWorld(config)
    board = sim_world.get_board()
    player = sim_world.get_player()

    # Initialize the Reinforcement Learner
    rl = ReinforcementLearner(sim_world.get_player(), config)
    actor = rl.get_actor()
    critic = rl.get_critic()

    # Log for remaining pegs at the end of each game
    remaining_pegs_pr_episode = []
    rewards = []

    # Start the generic actor-critic algorithm
    for episode in range(1, training_config["episodes"] + 1):
        if episode % 50 == 0:
            logging.info("Episode: {} - Avg Rewards: {}".format(episode, np.mean(rewards)))

        # If it is the last episode - no random actions should be selected
        if episode == training_config["episodes"]:
            logging.info("Epsilon = 0")
            actor.set_epsilon(0)
        else:
            actor.update_epsilon()

        # List of (state, action) pairs chosen this episode
        current_episode = []

        # Reset eligibility in actor and critic
        rl.reset_eligibility()

        # Initialize state and action
        state = board.to_binary_string_encoding()
        action = actor.get_action(state)
        while sim_world.is_neutral_state():
            # Legal actions in current state
            legal_actions = player.get_legal_actions()

            # Do action action from state, moving it to new_state and return reward
            new_state, reward = sim_world.perform_action(action)
            rewards.append(reward)

            # Get the action devoted to the new state by current policy
            new_action = actor.get_action(new_state)

            # Calculate the Temporal Difference error
            td_error = critic.get_td_error(state, new_state, reward)
            actor.set_td_error(td_error)

            # Set eligibility to 1 for both actor and critic
            rl.set_eligibility(state, action, is_current_state=True)

            # For all (state, action) pairs in this episode
            for s, a in current_episode:
                rl.update(s, a)

            # Save (state, action) to the "log" of this episode
            current_episode.append((state, action))

            # Continue until s reaches an end state
            state, action = new_state, new_action

        # Game ended, add results to log
        remaining_pegs_pr_episode.append(board.num_pegs_on_board())

        # Reset board for next game
        board.reset()

    # Report stats for debug
    actor.report_actor_stats()
    critic.report_critic_stats()

    # Visualize last episode
    sim_world.visualize_episode(current_episode, training_config)

    # All episodes has ran, save results
    plot_progression_of_learning(remaining_pegs_pr_episode, path=training_config["pol_path"])
