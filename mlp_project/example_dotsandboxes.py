import random
import pyspiel

states_dict = {}



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
    #state_key = state.dbn_string() + str(state.returns()[0])
    current_score1, current_score2 = get_current_score(state.__str__())
    state_key = state.dbn_string() + str(current_score1) + str(current_score2)
 
    

    if state.is_terminal():

        states_dict[state_key] = (-1,state.player_return(maximizing_player_id))
       
        return states_dict[state_key]
    
    if state_key in states_dict.keys():
    
     
        return states_dict[state_key]

    player = state.current_player()
    if player == maximizing_player_id:
        selection = max
    else:
        selection = min
    
    action_dict = {}
    #print("Legal actions inside minimax: ")
    #print(state.legal_actions())
    for action in state.legal_actions():
        (_,winner) = _minimax(state.child(action), maximizing_player_id)
        action_dict[action] = winner 

   
    
    best_action = selection(action_dict, key=action_dict.get)
    
    states_dict[state_key] = (best_action, action_dict[best_action])
    return states_dict[state_key]


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


    
    
    v = _minimax(
        state.clone(),
        maximizing_player_id=maximizing_player_id)
    return v





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
        action = minimax_search(game, state, self.player_id)[0]
        print("Test: " + str(action))
        return action


num_rows, num_cols = 2,3  # Number of squares
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
    state.apply_action(int(action))
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








import pyspiel
from absl import app










