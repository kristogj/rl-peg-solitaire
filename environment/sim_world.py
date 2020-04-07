from utils import get_board
from peg_solitaire_player import Player
from visualizer import BoardDrawer

import logging


class SimWorld:

    def __init__(self, config):
        logging.info("Setting up the Simulated World")
        self.board = get_board(config["Board"])  # Game board
        self.player = Player(self.board, config["Player"])  # Peq Solitaire Player
        self.visualizer = BoardDrawer(self.board, config["Training"])  # Class for visualizing board using networkx

    def is_winning_state(self):
        """
        If there is only one peg left on the board, the player has won
        :return: boolean
        """
        return self.board.num_pegs_on_board() == 1

    def is_loosing_state(self):
        """
        If there is more than one peg on the board, but no legal moves, you loose
        :return: boolean
        """
        return self.board.num_pegs_on_board() > 1 and len(self.player.get_legal_actions()) == 0

    def is_neutral_state(self):
        """
        If the board has more than one peg on the board, and at least one legal move, the game can still be played
        :return: boolean
        """
        return self.board.num_pegs_on_board() > 1 and len(self.player.get_legal_actions()) > 0

    def get_reward(self):
        """
        Return the reward of being in the boards state
        :return:
        """
        if self.is_winning_state():
            return 9999
        elif self.is_loosing_state():
            return - len(self.board.get_empty_cells()) ** 2
        else:
            return 0

    def get_player(self):
        """
        Return the Player of the game
        :return: Player
        """
        return self.player

    def get_board(self):
        """
        Return the Board of the game
        :return: Board
        """
        return self.board

    def visualize_episode(self, episode, config):
        board_drawer = BoardDrawer(self.board, config)
        for sap in episode:
            _, action = sap
            if action:
                board_drawer.draw(action=action)
                self.player.perform_action(action)
                board_drawer.draw()
        board_drawer.animate()
