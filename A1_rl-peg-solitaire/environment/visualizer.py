import networkx as nx
import matplotlib.pyplot as plt
import logging


# Visualize the Board using networkx
class BoardVisualizer:

    def __init__(self, board):
        logging.info("Initializing BoardVisualizer")
        self.board = board
        self.graph = self.gen_graph()
        self.positions = {cell: (cell.row, cell.column) for cell in self.board.get_cells()}

    def gen_graph(self):
        G = nx.Graph()
        for r in range(self.board.size):
            for c in range(self.board.size):
                cell = self.board.get_cell((r, c))
                if cell:
                    G.add_node(cell)
                    # Add all edges from cell
                neighbours = [(cell, neighbour) for neighbour in cell.neighbours.values()]
                G.add_edges_from(neighbours)

        return G

    def draw_board(self):
        # TODO: Not drawing correctly
        color_map = []
        for r in range(self.board.size):
            for c in range(self.board.size):
                cell = self.board.get_cell((r, c))
                if cell:
                    color = "blue" if cell.peg else "black"
                    color_map.append(color)
        print(self.graph.nodes)
        print(self.positions)
        print(color_map)
        nx.draw(self.graph, node_color=color_map, pos=self.positions)
        plt.show()
