from absl.testing import absltest
import numpy as np

from open_spiel.python import rl_environment, rl_tools
from open_spiel.python.algorithms.tabular_multiagent_qlearner import CorrelatedEqSolver
from open_spiel.python.algorithms.tabular_multiagent_qlearner import MultiagentQLearner
from open_spiel.python.algorithms.tabular_multiagent_qlearner import StackelbergEqSolver
from open_spiel.python.algorithms.tabular_multiagent_qlearner import TwoPlayerNashSolver
from open_spiel.python.algorithms.tabular_qlearner import QLearner
from open_spiel.python.egt.utils import game_payoffs_array
import pyspiel
import pyspiel
from open_spiel.python.algorithms import tabular_qlearner
from open_spiel.python.egt import dynamics
from open_spiel.python.egt import utils
from open_spiel.python.egt import visualization
from open_spiel.python.egt.utils import game_payoffs_array
import matplotlib.pyplot as plt


import numpy as np

game = pyspiel.create_matrix_game([[0, -0.05, 0.25], [0.05, 0, -0.5], [-0.25, 0.5, 0]], [[0, 0.05, -0.25], [-0.05, 0, 0.5], [0.25, -0.5, 0]])
env = rl_environment.Environment(game)
nashqlearner0 = MultiagentQLearner(0, 2,
                                    [env.game.num_distinct_actions()] * 2, step_size=0.001,epsilon_schedule=rl_tools.ConstantSchedule(1),
                                    joint_action_solver=TwoPlayerNashSolver())

nashqlearner1 = MultiagentQLearner(1, 2,
                                    [env.game.num_distinct_actions()] * 2, step_size=0.001,epsilon_schedule=rl_tools.ConstantSchedule(1),
                                    joint_action_solver=TwoPlayerNashSolver())


print("test")



payoff_tensor = utils.game_payoffs_array(env.game)
print(payoff_tensor)    
dyn = dynamics.SinglePopulationDynamics(payoff_tensor, dynamics.replicator)
x_list = np.array([[1/3, 1/3, 1/3], [0.2, 0.6, 0.2], [0.7, 0.1, 0.2], [0.05, 0.15, 0.8], [0.4, 0.3, 0.2]])


ax = plt.subplot(projection="3x3")
ax.quiver(dyn)

for i in range(1000):
    if (i % 100 == 0):
        print(i)
    time_step = env.reset()
    actions = [None, None]
    probs1 = nashqlearner0.step(time_step, actions, is_evaluation=True).probs
    probs2 = nashqlearner1.step(time_step, actions, is_evaluation=True).probs
    print(probs1)
    print(probs2)

    
    ax.scatter([(probs1[0], 
               probs1[1], probs1[2])],
               color="red", linestyle="dashed", linewidth=0.1)
    actions = [
        nashqlearner0.step(time_step, actions).action,
        nashqlearner1.step(time_step, actions).action
    ]
    time_step = env.step(actions)
    nashqlearner0.step(time_step, actions)
    nashqlearner1.step(time_step, actions)


plt.show()


time_step = env.reset()
actions = [None, None]
learner0_strategy, learner1_strategy = nashqlearner0.step(
    time_step, actions).probs, nashqlearner1.step(time_step,
                                                actions).probs

print(learner0_strategy)
print(learner1_strategy)
np.testing.assert_array_almost_equal(
    np.asarray([1 / 3, 1 / 3, 1 / 3]),
    learner0_strategy.reshape(-1),
    decimal=4)
np.testing.assert_array_almost_equal(
    np.asarray([1 / 3, 1 / 3, 1 / 3]),
    learner1_strategy.reshape(-1),
    decimal=4)


