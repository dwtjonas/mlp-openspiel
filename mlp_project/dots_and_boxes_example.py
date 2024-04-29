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

  game_string = "dots_and_boxes(num_rows=5,num_cols=5)"
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
  row = extract_dimensions(game_string)[0]
  column = extract_dimensions(game_string)[1]
  print(extract_dimensions(game_string))
  print('all the actions that are needed to close a single box')
  print((findActionsForBox(row,column)))
  print('actions that can close a single box')
  print(find_unique_numbers(findActionsForBox(row,column),state.legal_actions()))
  print('actions that give the opponet a box')
  print(find_actions_give_opponent_box(findActionsForBox(row,column),state.legal_actions()))
  print('actions that are already excecuted')
  print(taken_actions(init_actions(row,column),state.legal_actions()))
  print('should print all the actions to full in a half open chain')
  print(get_half_open_chains(findActionsForBox(row,column),state.legal_actions()))
  print('should print all the actions to full in a closed chain')
  print(get_closed_chains(findActionsForBox(row,column),state.legal_actions()))
  print('should print all the actions that can close in 2 boxes at once')
  print(hard_hearted_handout(findActionsForBox(row,column),state.legal_actions()))
  print('should print all the symmetries of the current state')
  print(symmetries(taken_actions(init_actions(row,column),state.legal_actions()),row,column))



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
    print(find_unique_numbers(findActionsForBox(row,column),state.legal_actions()))
    print('actions that give the opponet a box')
    print(find_actions_give_opponent_box(findActionsForBox(row,column),state.legal_actions()))
    print('actions that are already excecuted')
    print(taken_actions(init_actions(row,column),state.legal_actions()))
    print('should print all the actions to full in a half open chain')
    print(get_half_open_chains(findActionsForBox(row,column),state.legal_actions()))
    print('should print all the actions to full in a closed chain')
    print(get_closed_chains(findActionsForBox(row,column),state.legal_actions()))
    print('should print all the actions that can close in 2 boxes at once')
    print(hard_hearted_handout(findActionsForBox(row,column),state.legal_actions()))
    print('should print all the symmetries of the current state')
    print(symmetries(taken_actions(init_actions(row,column),state.legal_actions()),row,column))


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

def get_boxes_from_action(box_actions,number_action):
  result = []
  for box in box_actions:
    if number_action in box:
      result.append(box)
  return result

def get_half_open_chains(box_actions,actions):
   result = []
   subresult = []
   copy_actions = actions
   for action_close_box in find_unique_numbers(box_actions,actions):
      if (action_close_box not in copy_actions): # element can be removed by a chain that is not a solution
          copy_actions.append(action_close_box)
      copy_actions.remove(action_close_box)
      subresult.append(action_close_box)
      boxes = get_boxes_from_action(box_actions,action_close_box)
      while (len(boxes) == 2):
         flag = False
         subcount = 0
         for box in boxes:
            for element in box:
               if (element in copy_actions):
                   subcount += 1
            if (subcount == 1):
                for element in box:
                   if (element in copy_actions):
                      copy_actions.remove(element)
                      subresult.append(element)
                      boxes = get_boxes_from_action(box_actions,element)
                      flag = True
            if (subcount == 2 or subcount == 3):
                if (len(subresult) > 2):
                    result.append(subresult)
                    boxes = []
            subcount = 0 # reset
         if (not flag):
             boxes = [] # er wordt geen resultaat gevonden
      if (len(boxes) == 1 and len(subresult) > 2):
         result.append(subresult)
      subresult = []
   return result

def get_closed_chains(box_actions,actions):
   result = []
   subresult = []
   copy_actions = actions
   for action_close_box in find_unique_numbers(box_actions,actions):
      if (action_close_box not in copy_actions): # element can be removed by a chain that is not a solution
          copy_actions.append(action_close_box)
      copy_actions.remove(action_close_box)
      subresult.append(action_close_box)
      boxes = get_boxes_from_action(box_actions,action_close_box)
      while (len(boxes) == 2):
         flag = False
         subcount = 0
         for box in boxes: # check if both boxes are full
             for element in box:
                 if (element in copy_actions):
                     flag = True
         if (flag == False):
             if (len(subresult) > 1):
                 result.append(subresult)
             boxes = []
             continue
         flag = False
         for box in boxes:
            for element in box:
               if (element in copy_actions):
                   subcount += 1
            if (subcount == 1):
                for element in box:
                   if (element in copy_actions):
                      copy_actions.remove(element)
                      subresult.append(element)
                      boxes = get_boxes_from_action(box_actions,element)
                      flag = True
            subcount = 0 # reset
         if (not flag):
             boxes = []  # er wordt geen resultaat gevonden
      subresult = []
   return result

# get the actions that can fill in 2 boxes at once
def hard_hearted_handout(box_actions,actions):
    result = []
    number = 0
    count = 0
    for action in actions:
        boxes = get_boxes_from_action(box_actions, action)
        if len(boxes) == 2:
            for element in boxes[0]:
                if element in actions:
                    count += 1
                    number = element
            count = 0
            for element in boxes[1]:
                if element in actions:
                    count += 1
            if count == 1:
                result.append(number)
            count = 0
    return result

# task 3 point 2 symmetries

# output example ['h', 3, 1]
def givePositionLine(actionNumber,rowNumber,columnNumber):
    nb_hlines = (rowNumber + 1) * columnNumber
    if actionNumber < nb_hlines:
        row = actionNumber // columnNumber
        col = actionNumber % columnNumber
        return ['h',row,col]
    else:
        action2 = actionNumber - nb_hlines
        row = action2 // (columnNumber + 1)
        col = action2 % (columnNumber + 1)
        return ['v',row,col]

# output example is een innteger of the action
def giveActionLine(actionInfo, row_number, column_number):
    nb_hlines = (row_number + 1) * column_number
    direction, row, col = actionInfo
    if direction == 'h':
        action_number = row * column_number + col
    else:
        action_number = nb_hlines + row * (column_number + 1) + col
    return action_number



def equivalent_actions(actions, row_number, column_number):
    result = []
    subresult = []
    for action in actions:
        direction, row, col = action
        # horizontal mirroring
        if direction == 'h':
            subresult.append([direction, row_number - row, col])
        else:
            subresult.append([direction, row_number - row - 1, col])
    result.append(subresult)
    subresult = []
    for action in actions:
        direction, row, col = action
        # Vertical mirroring
        if direction == 'h':
            subresult.append([direction, row, column_number - col - 1])
        else:
            subresult.append([direction, row, column_number - col])
    result.append(subresult)
    subresult = []
    for action in actions:
        direction, row, col = action
        # first horizontal mirroring, then vertical mirroring or vice versa
        if direction == 'h':
            subresult.append([direction, row_number - row, column_number - col - 1])
        else:
            subresult.append([direction, row_number - row - 1, column_number - col])
    result.append(subresult)
    subresult = []
    if row_number == column_number:
        for action in actions:
            direction, row, col = action
            # first horizontal mirroring, then vertical mirroring or vice versa
            if direction == 'h':
                subresult.append(['v', col, row])
            else:
                subresult.append(['h', col, row])
        result.append(subresult)
        subresult = []
        for action in actions:
            direction, row, col = action
            # first horizontal mirroring, then vertical mirroring or vice versa
            if direction == 'h':
                subresult.append(['v', row_number - col - 1, row])
            else:
                subresult.append(['h', row_number - col, row])
        result.append(subresult)
        subresult = []
        for action in actions:
            direction, row, col = action
            # first horizontal mirroring, then vertical mirroring or vice versa
            if direction == 'h':
                subresult.append(['v', col, column_number - row])
            else:
                subresult.append(['h', col, column_number - row - 1])
        result.append(subresult)
        subresult = []
        for action in actions:
            direction, row, col = action
            # first horizontal mirroring, then vertical mirroring or vice versa
            if direction == 'h':
                subresult.append(['v', row_number - col - 1, column_number - row])
            else:
                subresult.append(['h', row_number - col, column_number - row - 1])
        result.append(subresult)
    return result

# output for example is [0, 2, 6]
def convertToActionNumbers(actionInfos,rowNumber,columnNumber):
    result = []
    subresult = []
    for element in actionInfos:
        for actionInfo in element:
            subresult.append(giveActionLine(actionInfo, rowNumber, columnNumber))
        subresult.sort()
        result.append(subresult)
        subresult = []
    return result

def symmetries(actionNumbers,rowNumber,columnNumber):
    result = []
    actionInfoList = []
    for actionNumber in actionNumbers:
        actionInfoList.append(givePositionLine(actionNumber,rowNumber,columnNumber))
    result = equivalent_actions(actionInfoList,rowNumber,columnNumber)
    return convertToActionNumbers(result,rowNumber,columnNumber)


if __name__ == "__main__":
  app.run(main)
