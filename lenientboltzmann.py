import pyspiel
from open_spiel.python import rl_environment, rl_tools
from open_spiel.python.algorithms import tabular_qlearner
from open_spiel.python.egt import dynamics
from open_spiel.python.egt import utils
from open_spiel.python.egt import visualization
from open_spiel.python.egt.utils import game_payoffs_array

import abc
import collections
import itertools
import nashpy as nash
import numpy as np

from open_spiel.python import rl_agent
from open_spiel.python import rl_tools
from open_spiel.python.algorithms.jpsro import _mgcce
from open_spiel.python.algorithms.stackelberg_lp import solve_stackelberg
import pyspiel


from open_spiel.python.algorithms.tabular_multiagent_qlearner import CorrelatedEqSolver, MultiagentQLearner, StackelbergEqSolver, TwoPlayerNashSolver, valuedict
import matplotlib.pyplot as plt


import numpy as np



class MultiagentBoltzmannQLearner(rl_agent.AbstractAgent):
  """A multiagent joint action learner."""

  def __init__(self,
               player_id,
               num_players,
               num_actions,
               joint_action_solver,
               step_size=0.1,
               epsilon_schedule=rl_tools.ConstantSchedule(0.2),
               discount_factor=1.0):
    """Initialize the Multiagent joint-action Q-Learning agent.

    The joint_action_solver solves for one-step matrix game defined by Q-tables.

    Args:
      player_id: the player id this agent will play as,
      num_players: the number of players in the game,
      num_actions: the number of distinct actions in the game,
      joint_action_solver: the joint action solver class to use to solve the
        one-step matrix games
      step_size: learning rate for Q-learning,
      epsilon_schedule: exploration parameter,
      discount_factor: the discount factor as in Q-learning.
    """
    self._player_id = player_id
    self._num_players = num_players
    self._num_actions = num_actions
    self._joint_action_solver = joint_action_solver
    self._step_size = step_size
    self._epsilon_schedule = epsilon_schedule
    self._epsilon = epsilon_schedule.value
    self._discount_factor = discount_factor
    #self._q_values = [
    #    collections.defaultdict(valuedict) for _ in range(num_players)
    #]
    self._q_values = [[1/3,1/3,1/3],[1/3,1/3,1/3]]
    
    self._prev_info_state = None

  def _get_payoffs_array(self, info_state):
    payoffs_array = np.zeros((self._num_players,) + tuple(self._num_actions))
    for joint_action in itertools.product(
        *[range(dim) for dim in self._num_actions]):
      for n in range(self._num_players):
        payoffs_array[
            (n,) + joint_action] = self._q_values[n][info_state][joint_action]
    return payoffs_array

  def _softmax(self, info_state, legal_actions, temperature):
    """Action selection based on boltzmann probability interpretation of Q-values.

    For more details, see equation (2) page 2 in
    https://arxiv.org/pdf/1109.1528.pdf

    Args:
        info_state: hashable representation of the information state.
        legal_actions: list of actions at `info_state`.
        temperature: temperature used for softmax.

    Returns:
        A valid soft-max selected action and valid action probabilities.
    """
    print(legal_actions)
    probs = np.zeros(self._num_actions)
    

    if temperature > 0.0:
      probs += [
          np.exp((1 / temperature) * self._q_values[self._player_id][i[self._player_id]])
          for i in range(self._num_actions)
      ]
      probs /= np.sum(probs)
    else:
      # Temperature = 0 causes normal greedy action selection
      greedy_q = max([self._q_values[self._player_id][a[self._player_id]] for a in legal_actions])
      greedy_actions = [
          a for a in legal_actions if self._q_values[self._player_id][a[self._player_id]] == greedy_q
      ]

      probs[greedy_actions] += 1 / len(greedy_actions)

    action = np.random.choice(range(self._num_actions), p=probs)
    return action, probs

  def step(self, time_step, actions=None, is_evaluation=False):
    """Returns the action to be taken and updates the Q-values if needed.

    Args:
        time_step: an instance of rl_environment.TimeStep,
        actions: list of actions taken by all agents from the previous step,
        is_evaluation: bool, whether this is a training or evaluation call,

    Returns:
        A `rl_agent.StepOutput` containing the action probs and chosen action.
    """
    info_state = str(time_step.observations["info_state"])
    legal_actions = time_step.observations["legal_actions"]

    # Prevent undefined errors if this agent never plays until terminal step
    action, probs = None, None

    # Act step: don't act at terminal states.
    if not time_step.last():
      epsilon = 0.0 if is_evaluation else self._epsilon
      # select according to the joint action solver
      action, probs = self._softmax(
          info_state, legal_actions, epsilon)

    # Learn step: don't learn during evaluation or at first agent steps.
    actions = tuple(actions)

    if self._prev_info_state and not is_evaluation:
      _, next_state_values = (
          self._joint_action_solver(self._get_payoffs_array(info_state)))
      # update Q values for every agent
      for n in range(self._num_players):
        target = time_step.rewards[n]
        if not time_step.last():  # Q values are zero for terminal.
          target += self._discount_factor * next_state_values[n]

        prev_q_value = self._q_values[n][self._prev_info_state][actions]

        self._q_values[n][self._prev_info_state][actions] += (
            self._step_size * (target - prev_q_value))

      # Decay epsilon, if necessary.
      self._epsilon = self._epsilon_schedule.step()

      if time_step.last():  # prepare for the next episode.
        self._prev_info_state = None
        return

    # Don't mess up with the state during evaluation.
    if not is_evaluation:
      self._prev_info_state = info_state

    return rl_agent.StepOutput(action=action, probs=probs)


game = pyspiel.create_matrix_game([[3, 0], [0, 2]], [[2, 0], [0, 3]])
env = rl_environment.Environment(game)

from open_spiel.python.algorithms import tabular_multiagent_qlearner

agents = [
    MultiagentBoltzmannQLearner(player_id=idx, num_players=2, num_actions=[env.game.num_distinct_actions()] * 2,
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
