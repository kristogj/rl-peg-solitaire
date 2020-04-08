import networkx as nx
import matplotlib.pyplot as plt
import math
from celluloid import Camera


class BoardVisualizer:
    """
    Class for visualizing all actions done during an episode of the game
    """
    def __init__(self, board, config):
        self.config = config
        self.board = board
        self.G = self.build_graph()
        self.positions = self.calculate_positions()
        self.fig = plt.figure()
        self.camera = Camera(self.fig)

    def build_graph(self):
        """
        Build the networkx graph with nodes and edges from the board
        :return: Graph
        """
        G = nx.Graph()
        for cell in self.board.get_cells():
            G.add_node(cell)
            # Add all edges from cell to its neighbours
            neighbours = [(cell, neighbour["cell"]) for neighbour in cell.get_neighbours()]
            G.add_edges_from(neighbours)
        return G

    def animate(self, path):
        """
        Animate each step to a gif
        """
        animation = self.camera.animate(
            repeat=False, interval=self.config['frame_delay'])
        animation.save(path)
        plt.show()

    def draw(self, action=None):
        """
        Draw the current state of the board
        """
        self.draw_occupied_cells()
        self.draw_open_cells()
        if action:
            self.draw_cell_peg_is_moving_from(action)
            self.draw_cell_peg_is_moving_to(action)
        self.draw_edges()
        plt.title('Peg Solitaire AI')
        self.camera.snap()

    def draw_occupied_cells(self):
        """
        Update which cells on the board that are pegs
        """
        pegs = [cell for cell in self.board.get_cells() if cell.is_peg]
        nx.draw_networkx_nodes(self.G, pos=self.positions, nodelist=pegs,
                               edgecolors='black', node_color='black', linewidths=2)

    def draw_open_cells(self):
        """
        Update which cells on the board that are empty
        """
        empty_cells = self.board.get_empty_cells()
        nx.draw_networkx_nodes(self.G, pos=self.positions, nodelist=empty_cells,
                               edgecolors='black', node_color='white', linewidths=2)

    def draw_cell_peg_is_moving_from(self, action):
        """
        Update which cell is the current from_ cell in an action
        """
        nx.draw_networkx_nodes(self.G, pos=self.positions, nodelist=[action.from_],
                               edgecolors='red', node_color='black', linewidths=2)

    def draw_cell_peg_is_moving_to(self, action):
        """
        Update which cell is the current to_ cell in an action
        """
        nx.draw_networkx_nodes(self.G, pos=self.positions, nodelist=[action.to_],
                               edgecolors='green', node_color='white', linewidths=2)

    def draw_edges(self):
        """
        Draw the edges of the board - showing which cell are neighbour with who
        """
        nx.draw_networkx_edges(self.G, pos=self.positions)

    def calculate_positions(self):
        """
        Calculate positions of each cell in the visualization
        """
        positions = {}
        row_number = 0
        for row in self.board.board:
            cell_number = 0
            number_of_cells = len(row)
            for cell in row:
                if row_number % 2 == 0:
                    xCoordinate = (number_of_cells / -2) + cell_number
                    yCoordinate = -row_number * math.sqrt(0.75)
                else:
                    xCoordinate = (number_of_cells / -2) + cell_number
                    yCoordinate = -row_number * math.sqrt(0.75)
                cell_number += 1
                positions[cell] = (xCoordinate, yCoordinate)
            row_number += 1
        return positions
