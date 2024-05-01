import random
import pyspiel
import dots_and_boxes_example as dab
import time
import numpy as np
states_dict = {}
num_rows, num_cols = 2,2  # Number of squares
chains_counter = 0


'''num_rows, num_cols = 2, 2  # Number of squares
game_string = (f"dots_and_boxes(num_rows={num_rows},num_cols={num_cols},"
                "utility_margin=true)")
game = pyspiel.load_game(game_string)
state = game.new_initial_state()
print(f"Initial state:")
print(state)
while not state.is_terminal():
    current_player = state.current_player()
    legal_actions = state.legal_actions()
    rand_idx = random.randint(0, len(legal_actions) - 1)
    action = legal_actions[rand_idx]
    state.apply_action(action)
    print(f"Player{current_player+1}:")

    print(state)
    print(state.returns())
    print(state.player_reward(0))
    print("Information string: " + state.information_state_string())
    #print("Information tensor: " + state.information_state_tensor())
    print("Is player node: " + str(state.is_player_node()))
    print("Observation string: " + state.observation_string())
    print("Observation string: " + state.observation_tensor())
    print("Player return : " + state.player_return(0))

    x = input()
returns = state.returns()
print(f"Player return values: {returns}")'''

def sign(n):
    if n > 0:
        return 1
    elif n == 0:
        return 0
    else:
        return -1


def get_current_score(observation_string):
    score_for_one = 0
    score_for_two = 0
    for c in observation_string:
        if c == '1':
            score_for_one += 1
        elif c == '2':
            score_for_two += 1
    return score_for_one, score_for_two

def get_agent_for_tournament(player_id):
    """Change this function to initialize your agent.
    This function is called by the tournament code at the beginning of the
    tournament.

    :param player_id: The integer id of the player for this bot, e.g. `0` if
        acting as the first player.
    """
    my_player = Agent(player_id)
    return my_player


def execute_list_of_actions(state, list_of_actions):
    if len(list_of_actions) == 0:
        return state
    print("state : ")
    print(state)
    print("list of actions: ")
    print(list_of_actions)
    return execute_list_of_actions(state.child(list_of_actions[0]), list_of_actions[1:])


def _minimax_action(state, maximizing_player_id):
    player = state.current_player()
    if player == 0:
        selection = max
    else:
        selection = min
    
    chain_actions = []
    half_open_chains = dab.get_half_open_chains(dab.findActionsForBox(num_rows, num_cols), state.legal_actions())
    closed_chains = dab.get_closed_chains(dab.findActionsForBox(num_rows, num_cols), state.legal_actions())
    '''for half_open_chain in half_open_chains:
        chain_actions.append(half_open_chain)
        print(half_open_chain)
        new_half_open_chain = half_open_chain[:]
        del new_half_open_chain[-2]
        print(new_half_open_chain)
        chain_actions.append(new_half_open_chain)
    action_dict = {}
    if len(half_open_chains) == 0:
        for action in state.legal_actions():
            action_dict[tuple([action])] = _minimax(state.child(action), maximizing_player_id) 
    else:
        global chains_counter
        chains_counter = chains_counter + 1
      
        for chain_action in chain_actions:
            new_state = execute_list_of_actions(state,chain_action)
            action_dict[tuple(chain_action)] = _minimax(new_state, maximizing_player_id)'''
    
    certain_actions_list = []
    possible_actions_list = []
    if len(half_open_chains) + len(closed_chains) >= 1:
        if len(half_open_chains) >= 1:
            
            for closed_chain in closed_chains:
                certain_actions_list.extend(closed_chain)
            for half_open_chain in half_open_chains[:-1]:
                certain_actions_list.extend(half_open_chain)

            half_open_chain = half_open_chains[-1]
            possible_actions_list.append(half_open_chain)
            new_half_open_chain = half_open_chain[:]
            del new_half_open_chain[-2]
            possible_actions_list.append(new_half_open_chain)
        else:
            for closed_chain in closed_chains[:-1]:
                certain_actions_list.extend(closed_chain)
            closed_chain = closed_chains[-1]
            possible_actions_list.append(closed_chain)
            new_closed_chain = closed_chain[:]
            del new_closed_chain[-2]
            if len(new_closed_chain) >= 4:
                del new_closed_chain[-4]
            possible_actions_list.append(new_closed_chain)


            
            
            

            
    #half_open_chain_actions = []
    #for half_open_chain in half_open_chains:
    #    half_open_chain_actions.append(half_open_chain)
    #    new_half_open_chain = half_open_chain[:]
    #    del new_half_open_chain[-2]
    #    half_open_chain_actions.append(new_half_open_chain)
    
    #closed_chains_actions = []

    #for closed_chain in closed_chains:
    #    closed_chains_actions.append(closed_chain)
    #    new_closed_chain = closed_chain[:]
    #    del new_closed_chain[-2]
    #    if len(new_closed_chain) >= 4:
    #        del new_closed_chain[-4]
    #    closed_chains_actions.append(new_closed_chain)


    action_dict = {}

    if len(half_open_chains) + len(closed_chains) == 0:
        for action in state.legal_actions():
            action_dict[tuple([action])] = _minimax(state.child(action), maximizing_player_id) 
    else:
        global chains_counter
        chains_counter = chains_counter + 1
    
        print("Certain actions list: ")
        print( certain_actions_list)
        new_state = execute_list_of_actions(state,certain_actions_list)

        for possible_action in possible_actions_list:
            print("Possible action: ")
            print(possible_action)
            newer_state = execute_list_of_actions(new_state,possible_action)
            action_dict[tuple(possible_action)]  = _minimax(newer_state, maximizing_player_id)
    

    best_action = selection(action_dict, key=action_dict.get)
    return best_action





def _minimax(state, maximizing_player_id):
    """
    Implements a min-max algorithm

    Arguments:
      state: The current state node of the game.
      maximizing_player_id: The id of the MAX player. The other player is assumed
        to be MIN.

    Returns:
      The optimal value of the sub-game starting in state
    """
    current_score1, current_score2 = get_current_score(state.__str__())
    state_key = state.dbn_string() + str(current_score1) + str(current_score2)

    if state.is_terminal():
        return state.player_return(maximizing_player_id)

    if state_key in states_dict.keys():
        return states_dict[state_key]

    player = state.current_player()
    if player == maximizing_player_id:
        selection = max
    else:
        selection = min

    half_open_chains = dab.get_half_open_chains(dab.findActionsForBox(num_rows, num_cols), state.legal_actions())
    closed_chains = dab.get_closed_chains(dab.findActionsForBox(num_rows, num_cols), state.legal_actions())
    
    certain_actions_list = []
    possible_actions_list = []
    if len(half_open_chains) + len(closed_chains) >= 1:
        if len(half_open_chains) >= 1:
            
            for closed_chain in closed_chains:
                certain_actions_list.extend(closed_chain)
            for half_open_chain in half_open_chains[:-1]:
                certain_actions_list.extend(half_open_chain)

            half_open_chain = half_open_chains[-1]
            possible_actions_list.append(half_open_chain)
            new_half_open_chain = half_open_chain[:]
            del new_half_open_chain[-2]
            possible_actions_list.append(new_half_open_chain)
        else:
            for closed_chain in closed_chains[:-1]:
                certain_actions_list.extend(closed_chain)
            closed_chain = closed_chains[-1]
            possible_actions_list.append(closed_chain)
            new_closed_chain = closed_chain[:]
            del new_closed_chain[-2]
            if len(new_closed_chain) >= 4:
                del new_closed_chain[-4]
            possible_actions_list.append(new_closed_chain)


            
            
            

            
    #half_open_chain_actions = []
    #for half_open_chain in half_open_chains:
    #    half_open_chain_actions.append(half_open_chain)
    #    new_half_open_chain = half_open_chain[:]
    #    del new_half_open_chain[-2]
    #    half_open_chain_actions.append(new_half_open_chain)
    
    #closed_chains_actions = []

    #for closed_chain in closed_chains:
    #    closed_chains_actions.append(closed_chain)
    #    new_closed_chain = closed_chain[:]
    #    del new_closed_chain[-2]
    #    if len(new_closed_chain) >= 4:
    #        del new_closed_chain[-4]
    #    closed_chains_actions.append(new_closed_chain)




    if len(half_open_chains) + len(closed_chains) == 0:
        values_children = [_minimax(state.child(action), maximizing_player_id) for action in state.legal_actions()]
        best_val = selection(values_children)
    else:
        global chains_counter
        chains_counter = chains_counter + 1
        state_list = []
        new_state = execute_list_of_actions(state,certain_actions_list)
        for possible_action in possible_actions_list:
            state_list.append(execute_list_of_actions(new_state,possible_action))
        values_children = [_minimax(state_list[i], maximizing_player_id) for i in range(len(state_list))]

        #if len(half_open_chains) == 0:
        #    for i in range(len(closed_chains_actions)):
        #        state_list.append(execute_list_of_actions(state,chain_actions[i]))
        #    values_children = [_minimax(state_list[i], maximizing_player_id) for i in range(len(state_list))]


        #for i in range(len(chain_actions)):
        #    state_list.append(execute_list_of_actions(state,chain_actions[i]))
      
        #values_children = [_minimax(state_list[i], maximizing_player_id) for i in range(len(state_list))]
    best_val = selection(values_children)

    #states_dict[state_key] = best_val

    for state_key in get_symmetrical_states(state_key): 
        states_dict[state_key] = best_val
    print(len(states_dict))


    return best_val

def minimax_search(game,
                   state=None,
                   maximizing_player_id=None,
                   state_to_key=lambda state: state):
    """Solves deterministic, 2-players, perfect-information 0-sum game.

    For small games only! Please use keyword arguments for optional arguments.

    Arguments:
      game: The game to analyze, as returned by `load_game`.
      state: The state to run from.  If none is specified, then the initial state is assumed.
      maximizing_player_id: The id of the MAX player. The other player is assumed
        to be MIN. The default (None) will suppose the player at the root to be
        the MAX player.

    Returns:
      The value of the game for the maximizing player when both player play optimally.
    """
    game_info = game.get_type()

    if game.num_players() != 2:
        raise ValueError("Game must be a 2-player game")
    if game_info.chance_mode != pyspiel.GameType.ChanceMode.DETERMINISTIC:
        raise ValueError("The game must be a Deterministic one, not {}".format(
            game.chance_mode))
    if game_info.information != pyspiel.GameType.Information.PERFECT_INFORMATION:
        raise ValueError(
            "The game must be a perfect information one, not {}".format(
                game.information))
    if game_info.dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL:
        raise ValueError("The game must be turn-based, not {}".format(
            game.dynamics))
    if game_info.utility != pyspiel.GameType.Utility.ZERO_SUM:
        raise ValueError("The game must be 0-sum, not {}".format(game.utility))

    if state is None:
        state = game.new_initial_state()
    if maximizing_player_id is None:
        maximizing_player_id = state.current_player()

    action = _minimax_action(
        state.clone(),
        maximizing_player_id=maximizing_player_id)
    return action

class Agent(pyspiel.Bot):
    """Agent template"""

    def __init__(self, player_id):
        """Initialize an agent to play Dots and Boxes.

        Note: This agent should make use of a pre-trained policy to enter
        the tournament. Initializing the agent should thus take no more than
        a few seconds.
        """
        pyspiel.Bot.__init__(self)
        self.player_id = player_id

    def restart_at(self, state):
        """Starting a new game in the given state.

        :param state: The initial state of the game.
        """
        pass

    def inform_action(self, state, player_id, action):
        """Let the bot know of the other agent's actions.

        :param state: The current state of the game.
        :param player_id: The ID of the player that executed an action.
        :param action: The action which the player executed.
        """
        pass

    def step(self, state):
        """Returns the selected action in the given state.

        :param state: The current state of the game.
        :returns: The selected action from the legal actions, or
            `pyspiel.INVALID_ACTION` if there are no legal actions available.
        """
        # Plays random action, change with your best strategy
        legal_actions = state.legal_actions()
        print("Legal actions: ")
        print(legal_actions)
        #rand_idx = random.randint(0, len(legal_actions) - 1)
        #action = legal_actions[rand_idx]

        tic = time.perf_counter()
        action = minimax_search(game, state, self.player_id)
        toc = time.perf_counter()
        print(f"Minimax in {toc - tic:0.4f} seconds")
        print("Test: " + str(action))

        return action


game_string = (f"dots_and_boxes(num_rows={num_rows},num_cols={num_cols},"
                "utility_margin=true)")
game = pyspiel.load_game(game_string)

bots = [get_agent_for_tournament(player_id) for player_id in [0,1]]

params = game.get_parameters()
assert num_rows == params['num_rows']
assert num_cols == params['num_cols']
num_cells = (num_rows + 1) * (num_cols + 1)
num_parts = 3   # (horizontal, vertical, cell)
num_states = 3  # (empty, player1, player2)

def part2num(part):
    p = {'h': 0, 'horizontal': 0,  # Who has set the horizontal line (top of cell)
         'v': 1, 'vertical':   1,  # Who has set the vertical line (left of cell)
         'c': 2, 'cell':       2}  # Who has won the cell
    return p.get(part, part)
def state2num(state):
    s = {'e':  0, 'empty':   0,
         'p1': 1, 'player1': 1,
         'p2': 2, 'player2': 2}
    return s.get(state, state)
def num2state(state):
    s = {0: 'empty', 1: 'player1', 2: 'player2'}
    return s.get(state, state)

print(f"{num_rows=}, {num_cols=}, {num_cells=}")



state = game.new_initial_state()
print(state)


current_player = state.current_player()
legal_actions = state.legal_actions()
print(legal_actions)

for action in legal_actions:
      print("Legal action: {} ({})".format(
          state.action_to_string(current_player, action), action))
      

action = 10
def action_transform(action):
    nb_hlines = (num_rows + 1) * num_cols
    if action < nb_hlines:
        row = action // num_cols
        col = action % num_cols
        print(f"{action=} : (h,{row},{col})")
    else:
        action2 = action - nb_hlines
        row = action2 // (num_cols + 1)
        col = action2 % (num_cols + 1)
        print(f"{action=} : (v,{row},{col})")


def get_observation(obs_tensor, state, row, col, part):
    state = state2num(state)
    part = part2num(part)
    idx =   part \
          + (row * (num_cols + 1) + col) * num_parts  \
          + state * (num_parts * num_cells)
    return obs_tensor[idx]
def get_observation_state(obs_tensor, row, col, part, as_str=True):
    is_state = None
    for state in range(3):
        if get_observation(obs_tensor, state, row, col, part) == 1.0:
            is_state = state
    if as_str:
        is_state = num2state(is_state)
    return is_state

def get_symmetrical_states(state_key):
    actions = [action for action, i in enumerate(state_key[:-2]) if i == '1']
    sym_actions = dab.symmetries(actions, num_rows, num_cols)
    sym_actions.append(actions)

    states = [state_key]
    for actions in sym_actions:
        sym_state_dbn = ''.join('1' if i in actions else '0' for i in range(len(state_key[:-2])))
        sym_state_key = sym_state_dbn + state_key[-2:]
        states.append(sym_state_key)    
    return states

'''state = game.new_initial_state()
state_strs, obs_tensors = [], []
actions = [0, 6]

state_strs += [f"{0:<{num_cols*4+1}}\n" + str(state)]
obs_tensors += [state.observation_tensor()]
for idx, action in enumerate(actions):
    state.apply_action(action)
    state_strs += [f"{idx+1:<{num_cols*4+1}}\n" + str(state)]
    obs_tensors += [state.observation_tensor()]

print("\n".join("   ".join(t) for t in zip(*[s.split("\n") for s in state_strs])))



(get_observation_state(obs_tensors[0], 0, 0, 'h'),
 get_observation_state(obs_tensors[2], 0, 0, 'h'))


(get_observation_state(obs_tensors[0], 0, 0, 'v'),
 get_observation_state(obs_tensors[2], 0, 0, 'v'))'''



state = game.new_initial_state()
print(f"Initial state:")
print(state)
while not state.is_terminal():
    current_player = state.current_player()
    
    legal_actions = state.legal_actions()
    #rand_idx = random.randint(0, len(legal_actions) - 1)
    #action = legal_actions[rand_idx]
    action = bots[current_player].step(state)
    print("Action: " + str(action))
    #action = current_player
    for line in action:
        state.apply_action(int(line))
    print(f"Player{current_player+1}:")
    print(state)
    
    '''if current_player == 1:
        legal_actions = state.legal_actions()
        print(legal_actions)
        action = int(input())
        print("Action: " + str(action))
        #action = current_player
        state.apply_action(int(action))
        print(f"Player{current_player+1}:")
        print(state)


    else:
    legal_actions = state.legal_actions()
    #rand_idx = random.randint(0, len(legal_actions) - 1)
    #action = legal_actions[rand_idx]
    action = bots[current_player].step(state)
    print("Action: " + str(action))
    #action = current_player
    state.apply_action(int(action))
    print(f"Player{current_player+1}:")
    print(state)'''
returns = state.returns()
print(f"Player return values: {returns}")
print()
print(chains_counter)








import pyspiel
from absl import app










