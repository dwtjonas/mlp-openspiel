import pyspiel
from open_spiel.python.algorithms import tabular_qlearner
from open_spiel.python.egt import dynamics
from open_spiel.python.egt import utils
from open_spiel.python.egt import visualization
from open_spiel.python.egt.utils import game_payoffs_array
import matplotlib.pyplot as plt


import numpy as np




game = pyspiel.create_matrix_game([[12, 0], [11, 10]], [[12, 11], [0, 10]])


agents = [
    tabular_qlearner.QLearner(player_id=idx, num_actions=2)
    for idx in range(2)
]


payoff_tensor = utils.game_payoffs_array(game)
print(payoff_tensor)    
dyn = dynamics.MultiPopulationDynamics(payoff_tensor, dynamics.replicator)
x_list = np.array([[0.5, 0.5, 0.5, 0.5], [0.1, 0.9, 0.1, 0.9], [0.8, 0.2, 0.9, 0.1], [0.1, 0.9, 0.7, 0.3], [0.85, 0.15, 0.17, 0.83]])


ax = plt.subplot(projection="2x2")
ax.quiver(dyn)


alpha = 0.01
for x in x_list:
    for i in range(10000):
        x += alpha * dyn(x)
        ax.scatter(x[0], x[2], color='red', linestyle='dashed', linewidth=0.1)


plt.show()






