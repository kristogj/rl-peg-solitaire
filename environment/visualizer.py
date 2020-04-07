import networkx as nx
import matplotlib.pyplot as plt
import math
from celluloid import Camera


class BoardDrawer:

    def __init__(self, board, config):
        self.config = config
        self.board = board
        self.G = self.build_graph()
        self.positions = self.calculate_postions()
        self.fig = plt.figure()
        self.camera = Camera(self.fig)

    def build_graph(self):
        G = nx.Graph()
        for cell in self.board.get_cells():
            G.add_node(cell)
            # Add all edges from cell to its neighbours
            neighbours = [(cell, neighbour["cell"]) for neighbour in cell.get_neighbours()]
            G.add_edges_from(neighbours)
        return G

    def animate(self):
        animation = self.camera.animate(
            repeat=False, interval=self.config['frame_delay'])
        animation.save("graphs/animation.gif")
        plt.show()

    def draw(self, action=None):

        self.draw_occupied_cells()
        self.draw_open_cells()
        if action:
            self.draw_cell_peg_is_moving_from(action)
            self.draw_cell_peg_is_moving_to(action)
        self.draw_edges()
        plt.title('Peg Solitaire AI')
        self.camera.snap()

    def draw_occupied_cells(self):
        occupied_cells = [cell for cell in self.board.get_cells() if cell.is_peg]
        nx.draw_networkx_nodes(self.G, pos=self.positions, nodelist=occupied_cells,
                               edgecolors='black', node_color='black', linewidths=2)

    def draw_open_cells(self):
        open_cells = self.board.get_empty_cells()
        nx.draw_networkx_nodes(self.G, pos=self.positions, nodelist=open_cells,
                               edgecolors='black', node_color='white', linewidths=2)

    def draw_cell_peg_is_moving_from(self, action):
        nx.draw_networkx_nodes(self.G, pos=self.positions, nodelist=[action.from_],
                               edgecolors='red', node_color='black', linewidths=2)

    def draw_cell_peg_is_moving_to(self, action):
        nx.draw_networkx_nodes(self.G, pos=self.positions, nodelist=[action.to_],
                               edgecolors='green', node_color='white', linewidths=2)

    def draw_edges(self):
        nx.draw_networkx_edges(self.G, pos=self.positions)

    def calculate_postions(self):
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
