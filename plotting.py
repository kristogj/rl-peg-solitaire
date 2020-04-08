import matplotlib.pyplot as plt
import numpy as np


def plot_progression_of_learning(remaining_pegs, path="./graphs/example_pol.png"):
    plt.title("Progression of Learning")
    plt.xlabel("Epochs")
    plt.ylabel("Remaining Pegs")
    episodes = np.arange(1, len(remaining_pegs) + 1)
    plt.plot(episodes, remaining_pegs)
    plt.savefig(path)
    plt.show()
