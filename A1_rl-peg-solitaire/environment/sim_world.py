from utils import get_board
from peg_solitaire_player import Player
from visualizer import BoardVisualizer

import logging


class SimWorld:

    def __init__(self, config):
        logging.info("Setting up the Simulated World")
        self.player = Player(config["Player"])  # Peq Solitaire Player
        self.board = get_board(config["Board"])  # Game board
        self.visualizer = BoardVisualizer(self.board)  # Class for visualizing board using networkx

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
        return self.board.num_pegs_on_board() > 1 and len(self.board.get_legal_actions()) == 0

    def is_neutral_state(self):
        """
        If the board has more than one peg on the board, and at least one legal move, the game can still be played
        :return: boolean
        """
        return self.board.num_pegs_on_board() > 1 and len(self.board.get_legal_actions()) > 0
