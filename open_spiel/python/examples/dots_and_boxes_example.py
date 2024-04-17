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
#
# Contributed by Wannes Meert, Giuseppe Marra, and Pieter Robberechts
# for the KU Leuven course Machine Learning: Project.


"""Python spiel example."""

from absl import app
from absl import flags
import numpy as np

from open_spiel.python.bots import human
from open_spiel.python.bots import uniform_random
import pyspiel

# added
from open_spiel.python.algorithms import minimax

FLAGS = flags.FLAGS

flags.DEFINE_integer("seed", 12761381, "The seed to use for the RNG.")

# Supported types of players: "random", "human"
flags.DEFINE_string("player0", "random", "Type of the agent for player 0.")
flags.DEFINE_string("player1", "random", "Type of the agent for player 1.")


def LoadAgent(agent_type, player_id, rng):
  """Return a bot based on the agent type."""
  if agent_type == "random":
    return uniform_random.UniformRandomBot(player_id, rng)
  elif agent_type == "human":
    return human.HumanBot()
  else:
    raise RuntimeError("Unrecognized agent type: {}".format(agent_type))


def main(_):
  rng = np.random.RandomState(FLAGS.seed)
  games_list = pyspiel.registered_names()
  assert "dots_and_boxes" in games_list

  game_string = "dots_and_boxes(num_rows=2,num_cols=2)"
  print("Creating game: {}".format(game_string))
  game = pyspiel.load_game(game_string)

  agents = [
      LoadAgent(FLAGS.player0, 0, rng),
      LoadAgent(FLAGS.player1, 1, rng),
  ]

  state = game.new_initial_state()

  # Print the initial state
  print("INITIAL STATE")
  print(str(state))
  if ((1) in state.legal_actions()):
    print(True)
  print(extract_dimensions(game_string))
  print('all the actions that are needed to close a single box')
  print((findActionsForBox(extract_dimensions(game_string)[0],extract_dimensions(game_string)[1])))
  print('actions that can close a single box')
  print(find_unique_numbers(findActionsForBox(2,2),state.legal_actions())) 
  print('actions that give the opponet a box')
  print(find_actions_give_opponent_box(findActionsForBox(2,2),state.legal_actions())) 
  print('actions that are already excecuted')
  print(taken_actions(init_actions(2,2),state.legal_actions()))

  while not state.is_terminal():
    current_player = state.current_player()
    # Decision node: sample action for the single current player
    legal_actions = state.legal_actions()
    for action in legal_actions:
      print(
          "Legal action: {} ({})".format(
              state.action_to_string(current_player, action), action
          )
      )
    action = agents[current_player].step(state)
    action_string = state.action_to_string(current_player, action)
    print("Player ", current_player, ", chose action: ", action_string)
    state.apply_action(action)

    print("")
    print("NEXT STATE:")
    print(str(state))
    print('actions that can close a single box')
    print(find_unique_numbers(findActionsForBox(2,2),state.legal_actions())) 
    print('actions that give the opponet a box')
    print(find_actions_give_opponent_box(findActionsForBox(2,2),state.legal_actions())) 
    print('actions that are already excecuted')
    print(taken_actions(init_actions(2,2),state.legal_actions()))
    if not state.is_terminal():
      print(str(state.observation_tensor()))

  # Game is now done. Print utilities for each player
  returns = state.returns()
  for pid in range(game.num_players()):
    print("Utility for player {} is {}".format(pid, returns[pid]))

# extract the information of the dimensions of a game
def extract_dimensions(game_string):
  return [int(game_string.split("=")[1][0]),int(game_string.split("=")[2][0])]

# given information about the field, return a tuple containing a list of all the number actions that close in a box.
def findActionsForBox(row, column):
    box_actions = []
    for i in range(row):
        for j in range(column):
            box_action = [
                i * column + j,
                i * column + j + column,
                i * column + j + (column*(row+1)) + i,
                i * column + j + (column*(row+1)) + i + 1
            ]
            box_actions.append(box_action)
    return tuple(box_actions)

# find the actions that can close in a box
# given al the actions that belongs to each other to close in a single box and all the available actions
def find_unique_numbers(box_actions,actions):
   subresult = []
   result = []
   for sublist in box_actions:
        for num in sublist:
            if ((num) in actions):
                subresult.append(num)
        if (len(subresult) == 1):
            result.append(subresult[0])
        subresult = []
   return result

# find the actions that if you draw it. It is the third line of a box and give basically a box to the opponent
# given al the actions that belongs to each other to close in a single box and all the available actions
def find_actions_give_opponent_box(box_actions,actions):
  subresult = []
  result = []
  for sublist in box_actions:
    for num in sublist:
      if ((num) in actions):
        subresult.append(num)
    if (len(subresult) == 2):
      result.append(subresult[0])
      result.append(subresult[1])
    subresult = []
  return result

# return wich all the possible actions in the beginning of the game
def init_actions(row,column):
  amount = row * (column + 1) + (row + 1) * column   # amount of actions at the start of the game
  return [i for i in range(amount)]

# return wich actions are not possible anymore. This can be used to detect chains.
def taken_actions(init_actions,actions):
  result = []
  for number in init_actions:
    if (number  not in actions):
      result.append(number)
  return result

# return the first initilal vetical line
def get_init_vertical_action(row,column):
   return (row + 1) * column
   
# return the coordinates(in form of a box) of a line
def get_coordinates_box(box_actions,action):
   return


if __name__ == "__main__":
  app.run(main)
