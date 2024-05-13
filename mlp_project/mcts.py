# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MCTS example."""

import collections
import random
import sys
import time
from absl import app
from absl import flags
import numpy as np

from open_spiel.python.algorithms import mcts
from open_spiel.python.algorithms.alpha_zero import evaluator as az_evaluator
from open_spiel.python.algorithms.alpha_zero import model as az_model
from open_spiel.python.bots import gtp
from open_spiel.python.bots import human
from open_spiel.python.bots import uniform_random
import pyspiel

import auxiliary_functions_task3 as dab # import dots_and_boxes_example as dab

num_rows = 7
num_cols = 7



_KNOWN_PLAYERS = [
    # A generic Monte Carlo Tree Search agent.
    "mcts",
    "mcts2",
    "mctswout",

    # A generic random agent.
    "random",

    # You'll be asked to provide the moves.
    "human",

    # Run an external program that speaks the Go Text Protocol.
    # Requires the gtp_path flag.
    "gtp",

    # Run an alpha_zero checkpoint with MCTS. Uses the specified UCT/sims.
    # Requires the az_path flag.
    "az2",
    "az"
]

flags.DEFINE_string("game", f"dots_and_boxes(num_rows={num_rows},num_cols={num_cols},"
                "utility_margin=true)", "Name of the game.")
flags.DEFINE_enum("player1", "mcts", _KNOWN_PLAYERS, "Who controls player 1.")
flags.DEFINE_enum("player2", "mcts2", _KNOWN_PLAYERS, "Who controls player 2.")
flags.DEFINE_enum("human", "human", _KNOWN_PLAYERS, "Who controls player 2.")

flags.DEFINE_string("gtp_path", None, "Where to find a binary for gtp.")
flags.DEFINE_multi_string("gtp_cmd", [], "GTP commands to run at init.")
flags.DEFINE_string("az_path", "./7x7new/checkpoint--1",
                    "Path to an alpha_zero checkpoint. Needed by an az player.")
flags.DEFINE_string("az_path2", "./woutcheckpoints/checkpoint--1",
                    "Path to an alpha_zero checkpoint. Needed by an az player.")
flags.DEFINE_integer("uct_c", 0, "UCT's exploration constant.")
flags.DEFINE_integer("rollout_count", 5, "How many rollouts to do.")
flags.DEFINE_integer("max_simulations", 10, "How many simulations to run.")
flags.DEFINE_integer("num_games", 25, "How many games to play.")
flags.DEFINE_integer("seed", None, "Seed for the random number generator.")
flags.DEFINE_bool("random_first", False, "Play the first move randomly.")
flags.DEFINE_bool("solve", True, "Whether to use MCTS-Solver.")
flags.DEFINE_bool("quiet", False, "Don't show the moves as they're played.")
flags.DEFINE_bool("verbose", False, "Show the MCTS stats of possible moves.")

FLAGS = flags.FLAGS
  
def get_current_score(observation_string):
    score_for_one = 0
    score_for_two = 0
    for c in observation_string:
        if c == '1':
            score_for_one += 1
        elif c == '2':
            score_for_two += 1
    return score_for_one, score_for_two

class WoutEvaluator:
    pass
    
  
class OwnEvaluator:
  
  def __init__(self, n_rollouts=1, random_state=None):
    self.n_rollouts = n_rollouts
    self._random_state = random_state or np.random.RandomState()

  def evaluate(self, state):
    """Returns evaluation on given state."""
    result = None
    working_state = state.clone()
    box_actions = dab.findActionsForBox(num_rows, num_cols)

    for _ in range(self.n_rollouts):  
      bonus_gain = 0    
      while not working_state.is_terminal():
        legal_actions =  working_state.legal_actions()
        half_open_chains = dab.get_half_open_chains(box_actions, legal_actions[:])
        #closed_chains = dab.get_closed_chains(box_actions, legal_actions[:])

        # Take all 3 sides squares that are not part of a half open chain
        three_sided = dab.find_unique_numbers(box_actions, legal_actions[:])
        for three_side in three_sided: 
            for half_open_chain in half_open_chains:
                if half_open_chain[0] == three_side: break
            working_state.apply_action(three_side)
            legal_actions.remove(three_side)
            #bonus_gain += 10

        if not working_state.is_terminal():
            half_open_chains = dab.get_half_open_chains(box_actions, legal_actions[:])
            if len(half_open_chains) >= 1:
                 # Take all
                if random.random() < 0.95:
                    for half_open_chain in half_open_chains:
                        execute_list_of_actions(working_state, half_open_chain)
                        #bonus_gain += 50

                # Leave last 2
                else:
                    half_open_chains = sorted(half_open_chains, key=len, reverse=True)
                    for half_open_chain in half_open_chains[:-1]:
                        execute_list_of_actions(working_state, half_open_chain)
                    last_half_open_chain = half_open_chains[-1][:]
                    del last_half_open_chain[-2]
                    execute_list_of_actions(working_state, last_half_open_chain)  
                    #bonus_gain += 10
      
              
        if not working_state.is_terminal():
            prefer_not_to = dab.find_actions_give_opponent_box(box_actions, legal_actions)
            if len(prefer_not_to) == len(legal_actions):
               action = self._random_state.choice(legal_actions)
            else:
               good_actions = [item for item in legal_actions if item not in prefer_not_to]
               action = self._random_state.choice(good_actions)
            working_state.apply_action(action)
    
      returns = np.array(working_state.returns())
      result = returns if result is None else result + returns

    return result / self.n_rollouts

  def prior(self, state):
    """Returns equal probability for all actions."""
    legal_actions = state.legal_actions(state.current_player())
    return [(action, 1.0 / len(legal_actions)) for action in legal_actions]
    
def execute_list_of_actions(state, list_of_actions):
    if len(list_of_actions) == 0:
        return state
    return execute_list_of_actions(state.child(list_of_actions[0]), list_of_actions[1:])
    
class RandomRolloutEvaluator:
  """A simple evaluator doing random rollouts.

  This evaluator returns the average outcome of playing random actions from the
  given state until the end of the game.  n_rollouts is the number of random
  outcomes to be considered.
  """

  def __init__(self, n_rollouts=1, random_state=None):
    self.n_rollouts = n_rollouts
    self._random_state = random_state or np.random.RandomState()

  def evaluate(self, state):
    """Returns evaluation on given state."""
    result = None
    for _ in range(self.n_rollouts):
      working_state = state.clone()
      while not working_state.is_terminal():
        action = self._random_state.choice(working_state.legal_actions())
        working_state.apply_action(action)
      returns = np.array(working_state.returns())
      result = returns if result is None else result + returns

    return result / self.n_rollouts

  def prior(self, state):
    """Returns equal probability for all actions."""
    if state.is_chance_node():
      return state.chance_outcomes()
    else:
      legal_actions = state.legal_actions(state.current_player())
      return [(action, 1.0 / len(legal_actions)) for action in legal_actions]

def _opt_print(*args, **kwargs):
  if not FLAGS.quiet:
    print(*args, **kwargs)


def _init_bot(bot_type, game, player_id):
  """Initializes a bot by type."""
  rng = np.random.RandomState(FLAGS.seed)
  if bot_type == "mcts":
    evaluator = RandomRolloutEvaluator(FLAGS.rollout_count, rng)
    return mcts.MCTSBot(
        game,
        FLAGS.uct_c,
        FLAGS.max_simulations,
        evaluator,
        random_state=rng,
        solve=FLAGS.solve,
        verbose=FLAGS.verbose)
  if bot_type == "mcts2":
    evaluator = OwnEvaluator(FLAGS.rollout_count, rng)
    return mcts.MCTSBot(
        game,
        FLAGS.uct_c,
        FLAGS.max_simulations,
        evaluator,
        random_state=rng,
        solve=FLAGS.solve,
        verbose=FLAGS.verbose)
  if bot_type == "mctswout":
    evaluator = WoutEvaluator(FLAGS.rollout_count, rng)
    return mcts.MCTSBot(
        game,
        FLAGS.uct_c,
        FLAGS.max_simulations,
        evaluator,
        random_state=rng,
        solve=FLAGS.solve,
        verbose=FLAGS.verbose)
  if bot_type == "az":
    model = az_model.Model.from_checkpoint(FLAGS.az_path)
    evaluator = az_evaluator.AlphaZeroEvaluator(game, model)
    return mcts.MCTSBot(
        game,
        FLAGS.uct_c,
        FLAGS.max_simulations,
        evaluator,
        random_state=rng,
        child_selection_fn=mcts.SearchNode.puct_value,
        solve=FLAGS.solve,
        verbose=FLAGS.verbose)
  if bot_type == "az2":
    model = az_model.Model.from_checkpoint(FLAGS.az_path2)
    evaluator = az_evaluator.AlphaZeroEvaluator(game, model)
    return mcts.MCTSBot(
        game,
        FLAGS.uct_c,
        FLAGS.max_simulations,
        evaluator,
        random_state=rng,
        child_selection_fn=mcts.SearchNode.puct_value,
        solve=FLAGS.solve,
        verbose=FLAGS.verbose)
  if bot_type == "random":
    return uniform_random.UniformRandomBot(player_id, rng)
  if bot_type == "human":
    return human.HumanBot()
  if bot_type == "gtp":
    bot = gtp.GTPBot(game, FLAGS.gtp_path)
    for cmd in FLAGS.gtp_cmd:
      bot.gtp_cmd(cmd)
    return bot
  raise ValueError("Invalid bot type: %s" % bot_type)


def _get_action(state, action_str):
  for action in state.legal_actions():
    if action_str == state.action_to_string(state.current_player(), action):
      return action
  return None


def _play_game(game, bots, initial_actions):
  """Plays one game."""
  state = game.new_initial_state()
  _opt_print("Initial state:\n{}".format(state))

  history = []

  if FLAGS.random_first:
    assert not initial_actions
    initial_actions = [state.action_to_string(
        state.current_player(), random.choice(state.legal_actions()))]

  for action_str in initial_actions:
    action = _get_action(state, action_str)
    if action is None:
      sys.exit("Invalid action: {}".format(action_str))

    history.append(action_str)
    for bot in bots:
      bot.inform_action(state, state.current_player(), action)
    state.apply_action(action)
    _opt_print("Forced action", action_str)
    _opt_print("Next state:\n{}".format(state))

  while not state.is_terminal():
    start_time = time.time()
    current_player = state.current_player()
    # Decision node: sample action for the single current player
    bot = bots[current_player]
    action = bot.step(state)
    action_str = state.action_to_string(current_player, action)
    _opt_print("Player {} sampled action: {}".format(current_player,
                                                    action_str))
    for i, bot in enumerate(bots):
      if i != current_player:
        bot.inform_action(state, current_player, action)
    history.append(action_str)
    state.apply_action(action)
    end_time = time.time()
    print("Move duration:" + str(end_time - start_time))
    _opt_print("Next state:\n{}".format(state))

  # Game is now done. Print return for each player
  returns = state.returns()
  print("Returns:", " ".join(map(str, returns)), ", Game actions:",
        " ".join(history))

  for bot in bots:
    bot.restart()

  return returns, history


def main(argv):
  game = pyspiel.load_game(FLAGS.game)
  if game.num_players() > 2:
    sys.exit("This game requires more players than the example can handle.")
  bots = [
      _init_bot(FLAGS.player1, game, 0),
      _init_bot(FLAGS.player2, game, 1),
  ]
  human = _init_bot(FLAGS.human, game, 1)
  histories = collections.defaultdict(int)
  overall_returns = [0, 0]
  overall_wins = [0, 0]
  game_num = 0
  try:
    for game_num in range(FLAGS.num_games):
      returns, history = _play_game(game, bots, argv[1:])
      histories[" ".join(history)] += 1
      for i, v in enumerate(returns):
        overall_returns[i] += v
        if v > 0:
          overall_wins[i] += 1
  except (KeyboardInterrupt, EOFError):
    game_num -= 1
    print("Caught a KeyboardInterrupt, stopping early.")
  print("Number of games played:", game_num + 1)
  print("Number of distinct games played:", len(histories))
  print("Players:", FLAGS.player1, FLAGS.player2)
  print("Overall wins", overall_wins)
  print("Overall returns", overall_returns)
  #_play_game(game, [bots[0], human], argv[1:])


if __name__ == "__main__":
  app.run(main)
