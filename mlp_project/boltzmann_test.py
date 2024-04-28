import pyspiel
import numpy as np
import matplotlib.pyplot as plt
from open_spiel.python import rl_environment
from open_spiel.python import rl_environment, rl_tools
from open_spiel.python.algorithms.boltzmann_tabular_qlearner import BoltzmannQLearner
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
import time

#test

'''from absl.testing import absltest
import numpy as np

from open_spiel.python import rl_environment
from open_spiel.python.algorithms import boltzmann_tabular_qlearner
import pyspiel

# Fixed seed to make test non stochastic.
SEED = 10000

# A simple two-action game encoded as an EFG game. Going left gets -1, going
# right gets a +1.
SIMPLE_EFG_DATA = """
  EFG 2 R "Simple single-agent problem" { "Player 1" } ""
  p "ROOT" 1 1 "ROOT" { "L" "R" } 0
    t "L" 1 "Outcome L" { -1.0 }
    t "R" 2 "Outcome R" { 1.0 }
"""


class BoltzmannQlearnerTest(absltest.TestCase):

  def test_simple_game(self):
    game = pyspiel.load_efg_game(SIMPLE_EFG_DATA)
    env = rl_environment.Environment(game=game)

    agent = boltzmann_tabular_qlearner.BoltzmannQLearner(
        0, game.num_distinct_actions())
    total_reward = 0

    for _ in range(100):
      total_eval_reward = 0
      for _ in range(1000):
        time_step = env.reset()
        while not time_step.last():
          agent_output = agent.step(time_step)
          time_step = env.step([agent_output.action])
          total_reward += time_step.rewards[0]
        agent.step(time_step)
      self.assertGreaterEqual(total_reward, 75)
      for _ in range(1000):
        time_step = env.reset()
        while not time_step.last():
          agent_output = agent.step(time_step, is_evaluation=True)
          time_step = env.step([agent_output.action])
          total_eval_reward += time_step.rewards[0]
      self.assertGreaterEqual(total_eval_reward, 250)


if __name__ == "__main__":
  np.random.seed(SEED)
  absltest.main()'''


#end test
game = pyspiel.create_matrix_game([[0, -0.05, 0.25], [0.05, 0, -0.5], [-0.25, 0.5, 0]], [[0, 0.05, -0.25], [-0.05, 0, 0.5], [0.25, -0.5, 0]])
env = rl_environment.Environment(game)

nashqlearner0 = BoltzmannQLearner(0, env.game.num_distinct_actions())

nashqlearner1 = BoltzmannQLearner(1, env.game.num_distinct_actions())

payoff_matrix = utils.game_payoffs_array(env.game)
dyn = dynamics.SinglePopulationDynamics(payoff_matrix, dynamics.replicator)

ax = plt.subplot(projection="3x3")
ax.quiver(dyn)


for i in range(1000):
    
    time_step = env.reset()
    #actions = np.array([None, None])

    probs1 = nashqlearner0.step(time_step,is_evaluation=True).probs
    probs2 = nashqlearner1.step(time_step,is_evaluation=True).probs
    print(probs1)
    print(probs2)
    ax.scatter([(probs1[0], probs1[1],probs1[2])], color="red", linestyle="dashed", linewidth=0.1)

   
    time_step = env.reset()
    agent0_output = nashqlearner0.step(time_step)
    agent1_output = nashqlearner1.step(time_step)
    #actions = np.array([None, None])
    #actions = np.array([
    #    nashqlearner0.step(time_step).action,
    #    nashqlearner1.step(time_step).action
    #])

    
    #time_step = env.step(np.array(actions))
    #info_state = str(time_step.observations["info_state"])
    #print(nashqlearner0._get_payoffs_array(info_state))
    time_step = env.step([agent0_output.action, agent1_output.action])
    nashqlearner0.step(time_step)
    nashqlearner1.step(time_step)
    #time.sleep(1)
    

plt.show()

time_step = env.reset()
#actions = [None, None]
#learner0_strategy, learner1_strategy = nashqlearner0.step(
#    time_step, actions).probs, nashqlearner1.step(time_step,
#                                                actions).probs

agent0_output = nashqlearner0.step(time_step, is_evaluation=True)
agent1_output = nashqlearner1.step(time_step, is_evaluation=True)
print(agent0_output)
print(agent1_output)

