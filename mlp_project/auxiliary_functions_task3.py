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
   return sorted(list(set(result)))#result

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
  return sorted(list(set(result)))#result

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

# get numbers from the side
def get_action_numbers_of_side(rowNumber,columNumber,box_actions):
    result = []
    for element in init_actions(rowNumber,columNumber):
        if len(get_boxes_from_action(box_actions,element)) == 1:
            result.append(element)
    return result


def is_sublist(lst1, lst2):
    """Controleert of lst1 een sublijst is van lst2."""
    n = len(lst1)
    m = len(lst2)

    # Als de lengte van lst1 groter is dan lst2, kan het geen sublijst zijn
    if n > m:
        return False

    # Zoek of lst1 een sublijst is van lst2
    for i in range(m - n + 1):
        if lst2[i:i + n] == lst1:
            return True

    return False

def remove_sublists(lst):
    """Verwijder sublijsten uit de lijst."""
    result = []
    for i in range(len(lst)):
        sub = lst[i]
        is_sub = False
        # Controleer of de sublijst een subverzameling is van een andere rij in de lijst
        for j in range(len(lst)):
            if i != j and is_sublist(sub, lst[j]):
                is_sub = True
                break
        # Voeg de sublijst toe aan het resultaat als het geen subverzameling is
        if not is_sub:
            result.append(sub)
    return result

# get all the actions that belong to an 2 way open chain
def get_both_open_chains(rowNumber,columnNumber,box_actions, actions):
    sideActions = init_actions(rowNumber,columnNumber)#get_action_numbers_of_side(rowNumber,columnNumber,box_actions)
    copy_actions = actions.copy()
    count = 0
    result = []
    subresult = []
    nextNumber = 0
    for element in sideActions:
        copy_actions = actions.copy()
        if element not in copy_actions: # if element not in actions, we don't need to do the search
            continue
        if (element not in copy_actions):  # element can be removed by a chain that is not a solution
            copy_actions.append(element)
        copy_actions.remove(element)
        boxes = get_boxes_from_action(box_actions,element)
        box = boxes[0]
        for number in box:
            if number in copy_actions:
                count += 1
                nextNumber = number
        if count == 1:
            subresult.append(element)
        boxes = get_boxes_from_action(box_actions, nextNumber)
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
                            boxes = get_boxes_from_action(box_actions, element)
                            flag = True
                if (subcount == 2 or subcount == 3):
                    if (len(subresult) > 3):
                        result.append(subresult)
                        boxes = []
                subcount = 0  # reset
            if (not flag):
                boxes = []  # er wordt geen resultaat gevonden
        if (len(boxes) == 1 and len(subresult) > 3):
            result.append(subresult)
        subresult = []
        count = 0
    result = remove_sublists(result)
    resultNoDup = []
    for sublist in result:
        if sublist[::-1] not in resultNoDup:
            resultNoDup.append(sublist)
    return resultNoDup

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