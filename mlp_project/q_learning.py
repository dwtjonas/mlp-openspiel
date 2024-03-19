# First attempt at Q-learning in OpenSpiel (doesn’t work yet)
import collections
import pyspiel
from open_spiel.python import rl_environment, rl_tools
from open_spiel.python.algorithms import tabular_qlearner
from open_spiel.python.algorithms import tabular_multiagent_qlearner
from open_spiel.python.egt import dynamics
from open_spiel.python.egt import utils
from open_spiel.python.egt import visualization
from open_spiel.python.algorithms.tabular_multiagent_qlearner import CorrelatedEqSolver, MultiagentQLearner, StackelbergEqSolver, TwoPlayerNashSolver, valuedict
from open_spiel.python.egt.utils import game_payoffs_array
import matplotlib.pyplot as plt


import numpy as np


# ik ben wout

print("This file has been changed")
game = pyspiel.load_game("matrix_bos")
state = game.new_initial_state()
env = rl_environment.Environment("matrix_bos")
num_players = env.num_players
num_actions = env.action_spec()["num_actions"]

nashqlearner0 = MultiagentQLearner(0, 2,
                                    [env.game.num_distinct_actions()] * 2,
                                    TwoPlayerNashSolver(), epsilon_schedule=rl_tools.LinearSchedule(1, 0.1, 250))

nashqlearner1 = MultiagentQLearner(1, 2,
                                    [env.game.num_distinct_actions()] * 2,
                                    TwoPlayerNashSolver(), epsilon_schedule=rl_tools.LinearSchedule(1, 0.1, 250))

payoff_tensor = utils.game_payoffs_array(game)
print(payoff_tensor)    
dyn = dynamics.MultiPopulationDynamics(payoff_tensor, dynamics.replicator)
ax = plt.subplot(projection="2x2")
ax.quiver(dyn)

for i in range(250):
    time_step = env.reset()
    actions = [None, None]

    learner0_strategy, learner1_strategy = (
        nashqlearner0.step(time_step, actions, is_evaluation=True).probs, 
        nashqlearner1.step(time_step, actions, is_evaluation=True).probs
    )

    ax.scatter(learner0_strategy[0], learner1_strategy[0], color='red', linestyle='dashed', linewidth=0.1)
    if i % 50 == 0:
        print("Iteration:", i)
        print(learner0_strategy)
        print(learner1_strategy)
    actions = [
        nashqlearner0.step(time_step, actions).action,
        nashqlearner1.step(time_step, actions).action
    ]
    time_step = env.step(actions)
    nashqlearner0.step(time_step, actions)
    nashqlearner1.step(time_step, actions)


plt.show()




'''
game = pyspiel.create_matrix_game([[3, 0], [0, 2]], [[2, 0], [0, 3]])

state = game.new_initial_state()

env = rl_environment.Environment(game)

agents = [
    tabular_multiagent_qlearner.MultiagentQLearner(player_id=idx, num_players=2, num_actions=[env.game.num_distinct_actions()] * 2, joint_action_solver=TwoPlayerNashSolver)
    for idx in range(num_players)
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
        ax.scatter(x[0], x[2], color='red', linestyle='dashed', linewidth=0.1)


plt.show()"""

env = rl_environment.Environment("matrix_rps")
nashqlearner0 = tabular_multiagent_qlearner.MultiagentQLearner(0, 2,
                                       [env.game.num_distinct_actions()] * 2,
                                       TwoPlayerNashSolver())


nashqlearner1 = tabular_multiagent_qlearner.MultiagentQLearner(1, 2,
                                       [env.game.num_distinct_actions()] * 2,
                                       TwoPlayerNashSolver())


"""for _ in range(1000):
    time_step = env.reset()
    actions = [None, None]
    actions = [
        nashqlearner0.step(time_step, actions).action,
        nashqlearner1.step(time_step, actions).action
    ]
    time_step = env.step(actions)
    nashqlearner0.step(time_step, actions)
    nashqlearner1.step(time_step, actions)


learner0_strategy, learner1_strategy = nashqlearner0.step(
          time_step, actions).probs, nashqlearner1.step(time_step,
                                                        actions).probs


print(learner0_strategy)
print(learner1_strategy)"""

for i in range(1000):
    time_step = env.reset()  
    actions = [None, None]
    actions = [
        agents[0].step(time_step, actions).action,
        agents[1].step(time_step, actions).action
    ]
    time_step = env.step(actions)
    agents[0].step(time_step, actions)
    agents[1].step(time_step, actions)


    time_step = env.reset()
strategies = [agents[0].step(time_step, actions, is_evaluation=False).probs, agents[1].step(time_step, actions, is_evaluation=False).probs]
print(strategies)
'''