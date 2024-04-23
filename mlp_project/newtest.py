import pyspiel
import numpy as np
import matplotlib.pyplot as plt
from open_spiel.python import rl_environment
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


game = pyspiel.create_matrix_game([[-1, -4], [0, -3]], [[-1, 0], [-4, -3]])
env = rl_environment.Environment(game)

nashqlearner0 = MultiagentQLearner(0, 2,
                                    [env.game.num_distinct_actions()] * 2, step_size=0.1,epsilon_schedule=rl_tools.LinearSchedule(1,0,1000),
                                    joint_action_solver=TwoPlayerNashSolver())

nashqlearner1 = MultiagentQLearner(1, 2,
                                    [env.game.num_distinct_actions()] * 2, step_size=0.1,epsilon_schedule=rl_tools.LinearSchedule(1,0,1000),
                                    joint_action_solver=TwoPlayerNashSolver())

payoff_matrix = utils.game_payoffs_array(env.game)
dyn = dynamics.MultiPopulationDynamics(payoff_matrix, dynamics.replicator)

ax = plt.subplot(projection="2x2")
ax.quiver(dyn)



for i in range(1000):
    
    time_step = env.reset()
    actions = np.array([None, None])

    probs1 = nashqlearner0.step(time_step, actions, is_evaluation=True).probs
    probs2 = nashqlearner1.step(time_step, actions, is_evaluation=True).probs
    #print(probs1)
    #print(probs2)
    ax.scatter(probs1[0], probs2[0], color="red", linestyle="dashed", linewidth=0.1)

   
    time_step = env.reset()
    actions = np.array([None, None])
    actions = np.array([
        nashqlearner0.step(time_step, actions).action,
        nashqlearner1.step(time_step, actions).action
    ])

    
    time_step = env.step(np.array(actions))
    info_state = str(time_step.observations["info_state"])
    print(nashqlearner0._get_payoffs_array(info_state))
    nashqlearner0.step(time_step, actions)
    nashqlearner1.step(time_step, actions)

plt.show()

time_step = env.reset()
#actions = [None, None]
learner0_strategy, learner1_strategy = nashqlearner0.step(
    time_step, actions, is_evaluation=True).probs, nashqlearner1.step(time_step,
                                                actions, is_evaluation=True).probs
print(learner0_strategy, learner1_strategy)