# Rock paper scissors
import pyspiel
from open_spiel.python.algorithms import tabular_qlearner
from open_spiel.python.egt import dynamics
from open_spiel.python.egt import utils
from open_spiel.python.egt import visualization
from open_spiel.python.egt.utils import game_payoffs_array
import matplotlib.pyplot as plt


import numpy as np




game = pyspiel.create_matrix_game([[0, -0.05, 0.25], [0.05, 0, -0.5], [-0.25, 0.5, 0]], [[0, 0.05, -0.25], [-0.05, 0, 0.5], [0.25, -0.5, 0]])


agents = [
    tabular_qlearner.QLearner(player_id=idx, num_actions=2)
    for idx in range(2)
]


payoff_tensor = utils.game_payoffs_array(game)
print(payoff_tensor)    
dyn = dynamics.SinglePopulationDynamics(payoff_tensor, dynamics.replicator)
x_list = np.array([[1/3, 1/3, 1/3], [0.2, 0.6, 0.2], [0.7, 0.1, 0.2], [0.05, 0.15, 0.8], [0.4, 0.3, 0.2]])


ax = plt.subplot(projection="3x3")
ax.quiver(dyn)


alpha = 0.01
for x in x_list:
    for i in range(2000):
        x += alpha * dyn(x)
        ax.scatter([(x[0], x[1], x[2])], color='red', linestyle='dashed', linewidth=0.1)


plt.show()
