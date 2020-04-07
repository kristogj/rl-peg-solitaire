import matplotlib.pyplot as plt
import networkx
import numpy as np


def plot_progression_of_learning(remaining_pegs, path="./graphs/progression_of_learning.png"):
    plt.title("Progression of Learning")
    plt.xlabel("Epochs")
    plt.ylabel("Remaining Pegs")
    episodes = np.arange(1, len(remaining_pegs) + 1)
    plt.plot(episodes, remaining_pegs)
    plt.savefig(path)
    plt.show()


def display_moves():
    """
    Use Pythons networkx. See Figure 2 of assignment
    https://networkx.github.io/documentation/stable/
    """
    pass


def test():
    epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    pegs = [10, 10, 10, 9, 9, 8, 7, 7, 7, 6]
    plot_progression_of_learning(epochs, pegs)

# test()
