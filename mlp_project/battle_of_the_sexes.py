# unfinished
import pyspiel
from open_spiel.python import rl_environment, rl_tools
from open_spiel.python.algorithms import tabular_qlearner
from open_spiel.python.egt import dynamics
from open_spiel.python.egt import utils
from open_spiel.python.egt import visualization
from open_spiel.python.egt.utils import game_payoffs_array

from open_spiel.python.algorithms.tabular_multiagent_qlearner import CorrelatedEqSolver, MultiagentQLearner, StackelbergEqSolver, TwoPlayerNashSolver, valuedict
import matplotlib.pyplot as plt


import numpy as np




game = pyspiel.create_matrix_game([[3, 0], [0, 2]], [[2, 0], [0, 3]])
env = rl_environment.Environment(game)

from open_spiel.python.algorithms import tabular_multiagent_qlearner

agents = [
    tabular_multiagent_qlearner.MultiagentQLearner(player_id=idx, num_players=2, num_actions=[env.game.num_distinct_actions()] * 2,
                                 joint_action_solver=TwoPlayerNashSolver(), epsilon_schedule=rl_tools.ConstantSchedule(0.9), step_size=0.1)
    for idx in range(2)
]


payoff_tensor = utils.game_payoffs_array(game)
print(payoff_tensor)    
dyn = dynamics.MultiPopulationDynamics(payoff_tensor, dynamics.replicator)
x_list = np.array([[0.5, 0.5, 0.5, 0.5], [0.1, 0.9, 0.1, 0.9], [0.8, 0.2, 0.9, 0.1], [0.1, 0.9, 0.7, 0.3], [0.85, 0.15, 0.17, 0.83]])


ax = plt.subplot(projection="2x2")
ax.quiver(dyn)


"""alpha = 0.01
for x in x_list:
    for i in range(10000):
        x += alpha * dyn(x)
        ax.scatter(x[0], x[2], color='red', linestyle='dashed', linewidth=0.1)"""

for i in range(1000):
    if (i % 100 == 0):
        print(i)
    time_step = env.reset()
    actions = [None, None]
    if i % 106 == 0:
        probs1 = agents[0].step(time_step, actions, is_evaluation=True).probs
        probs2 = agents[1].step(time_step, actions, is_evaluation=True).probs
        print(probs1)
        print(probs2)

        
        ax.scatter(probs1, probs2,
                color="red", linestyle="dashed", linewidth=0.1)
    actions = [
        agents[0].step(time_step, actions).action,
        agents[1].step(time_step, actions).action
    ]
    time_step = env.step(actions)
    agents[0].step(time_step, actions)
    agents[1].step(time_step, actions)


plt.show()





