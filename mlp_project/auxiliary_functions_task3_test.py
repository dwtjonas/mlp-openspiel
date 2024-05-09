import unittest

from auxiliary_functions_task3 import (
    extract_dimensions,
    findActionsForBox,
    find_unique_numbers,
    find_actions_give_opponent_box,
    init_actions,
    taken_actions,
    get_init_vertical_action,
    get_coordinates_box,
    get_boxes_from_action,
    get_half_open_chains,
    get_closed_chains,
    hard_hearted_handout,
    get_action_numbers_of_side,
    is_sublist,
    remove_sublists,
    get_both_open_chains,
    givePositionLine,
    giveActionLine,
    equivalent_actions,
    convertToActionNumbers,
    symmetries,
)

class TestFunctions(unittest.TestCase):

    def test_extract_dimensions(self):
        game_string = "dots_and_boxes(num_rows=3,num_cols=3)"
        self.assertEqual(extract_dimensions(game_string), [3,3])
        game_string = "dots_and_boxes(num_rows=2,num_cols=2)"
        self.assertEqual(extract_dimensions(game_string), [2,2])

    def test_findActionsForBox(self):
        row = 3
        column = 3
        self.assertEqual(findActionsForBox(row,column), ([0, 3, 12, 13], [1, 4, 13, 14], [2, 5, 14, 15], [3, 6, 16, 17], [4, 7, 17, 18], [5, 8, 18, 19], [6, 9, 20, 21], [7, 10, 21, 22], [8, 11, 22, 23]))
        row = 2
        column = 2
        self.assertEqual(findActionsForBox(row,column),([0, 2, 6, 7], [1, 3, 7, 8], [2, 4, 9, 10], [3, 5, 10, 11]))
        row = 2
        column = 3
        self.assertEqual(findActionsForBox(row,column),([0, 3, 9, 10], [1, 4, 10, 11], [2, 5, 11, 12], [3, 6, 13, 14], [4, 7, 14, 15], [5, 8, 15, 16]))

    def test_find_unique_numbers(self):
        box_actions = [[0, 2, 6, 7], [1, 3, 7, 8], [2, 4, 9, 10], [3, 5, 10, 11]]
        actions = [4, 5, 7, 11]
        self.assertEqual(find_unique_numbers(box_actions,actions),[4,7])
        box_actions = [[0, 2, 6, 7], [1, 3, 7, 8], [2, 4, 9, 10], [3, 5, 10, 11]]
        actions = [4, 6, 7, 8]
        self.assertEqual(find_unique_numbers(box_actions,actions),[4])

    def test_find_actions_give_opponent_box(self):
        box_actions = [[0, 2, 6, 7], [1, 3, 7, 8], [2, 4, 9, 10], [3, 5, 10, 11]]
        actions = [2, 3, 7, 10]
        self.assertEqual(find_actions_give_opponent_box(box_actions,actions),[2, 3, 7, 10])

    def test_init_actions(self):
        row = 3
        column = 3
        self.assertEqual(init_actions(row, column), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])
        row = 2
        column = 2
        self.assertEqual(init_actions(row, column), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

    def test_taken_actions(self):
        row = 3
        column = 3
        actions = [2, 3, 7, 10]
        self.assertEqual(taken_actions(init_actions(row,column), actions), [0, 1, 4, 5, 6, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])

    def test_get_boxes_from_action(self):
        box_actions = [[0, 3, 12, 13], [1, 4, 13, 14], [2, 5, 14, 15], [3, 6, 16, 17], [4, 7, 17, 18], [5, 8, 18, 19],
               [6, 9, 20, 21], [7, 10, 21, 22], [8, 11, 22, 23]]
        number_action = 5
        self.assertEqual(sorted(get_boxes_from_action(box_actions,number_action)),sorted([[2, 5, 14, 15], [5, 8, 18, 19]]))

    def test_get_half_open_chains(self):
        box_actions = [[0, 3, 12, 13], [1, 4, 13, 14], [2, 5, 14, 15], [3, 6, 16, 17], [4, 7, 17, 18], [5, 8, 18, 19], [6, 9, 20, 21], [7, 10, 21, 22], [8, 11, 22, 23]]
        actions = [5, 6, 7, 8, 9, 10, 11, 13, 14, 16, 17, 20, 21, 22, 23]
        self.assertEqual(get_half_open_chains(box_actions,actions),[[13, 14, 5, 8]])
        box_actions = [[0, 3, 12, 13], [1, 4, 13, 14], [2, 5, 14, 15], [3, 6, 16, 17], [4, 7, 17, 18], [5, 8, 18, 19], [6, 9, 20, 21], [7, 10, 21, 22], [8, 11, 22, 23]]
        actions = [1, 3, 5, 6, 7, 10, 11, 13, 18, 23]
        self.assertEqual(sorted(get_half_open_chains(box_actions,actions)),sorted([[5, 18, 7, 10], [6, 3, 13, 1]]))
        box_actions = [[0, 2, 6, 7], [1, 3, 7, 8], [2, 4, 9, 10], [3, 5, 10, 11]]
        actions = [2, 5, 7, 8, 11]
        self.assertEqual(sorted(get_half_open_chains(box_actions,actions)),sorted([[2, 7, 8]]))
        box_actions = [[0, 3, 12, 13], [1, 4, 13, 14], [2, 5, 14, 15], [3, 6, 16, 17], [4, 7, 17, 18], [5, 8, 18, 19], [6, 9, 20, 21], [7, 10, 21, 22], [8, 11, 22, 23]]
        actions = [6, 7, 8, 11, 13, 14, 18, 21]
        self.assertEqual(sorted(get_half_open_chains(box_actions,actions)),sorted([[6, 21, 7, 18, 8, 11]]))

    def test_get_closed_chains(self):
        box_actions = [[0, 2, 6, 7], [1, 3, 7, 8], [2, 4, 9, 10], [3, 5, 10, 11]]
        actions = [3, 4, 7, 9]
        self.assertEqual(get_closed_chains(box_actions,actions),[[3, 7]])
        box_actions = [[0, 3, 12, 13], [1, 4, 13, 14], [2, 5, 14, 15], [3, 6, 16, 17], [4, 7, 17, 18], [5, 8, 18, 19], [6, 9, 20, 21], [7, 10, 21, 22], [8, 11, 22, 23]]
        actions = [6, 7, 8, 13, 14, 18, 21]
        self.assertEqual(get_closed_chains(box_actions,actions),[[6, 21, 7, 18, 8], [13, 14]])

    def test_hard_hearted_handout(self):
        box_actions = [[0, 2, 6, 7], [1, 3, 7, 8], [2, 4, 9, 10], [3, 5, 10, 11]]
        actions = [7, 10]
        self.assertEqual(hard_hearted_handout(box_actions,actions),[7,10])

    def test_get_action_numbers_of_side(self):
        rowNumber = 3
        columNumber = 3
        box_actions = [[0, 3, 12, 13], [1, 4, 13, 14], [2, 5, 14, 15], [3, 6, 16, 17], [4, 7, 17, 18], [5, 8, 18, 19],
               [6, 9, 20, 21], [7, 10, 21, 22], [8, 11, 22, 23]]
        self.assertEqual(get_action_numbers_of_side(rowNumber,columNumber,box_actions),[0, 1, 2, 9, 10, 11, 12, 15, 16, 19, 20, 23])

    def test_is_sublist(self):
        self.assertFalse(is_sublist([4, 3], [1, 2, 3, 4, 5]))
        self.assertTrue(is_sublist([1, 2, 3], [1, 2, 3, 4, 5]))

    def test_remove_sublists(self):
        self.assertEqual(remove_sublists([[1, 2], [2, 3], [4, 5], [6, 7], [2, 3, 4]]), [[1, 2], [4, 5], [6, 7], [2, 3, 4]])

    def test_givePositionLine(self):
        actionNumber = 10
        rowNumber = 3
        columnNumber = 3
        self.assertEqual(givePositionLine(actionNumber,rowNumber,columnNumber),['h', 3, 1])

    def test_giveActionLine(self):
        actionInfo = ['h', 2, 0]
        row_number = 3
        column_number = 3
        self.assertEqual(giveActionLine(actionInfo, row_number, column_number),6)
    
    def test_equivalent_actions(self):
        actions = [['h', 0, 0]]
        row_number = 3
        column_number = 3
        self.assertEqual(equivalent_actions(actions, row_number, column_number),[[['h', 3, 0]], [['h', 0, 2]], [['h', 3, 2]], [['v', 0, 0]], [['v', 2, 0]], [['v', 0, 3]], [['v', 2, 3]]])
        actions = [['h', 2, 0],['h', 0, 2]]
        row_number = 3
        column_number = 3
        self.assertEqual(equivalent_actions(actions, row_number, column_number),[[['h', 1, 0], ['h', 3, 2]], [['h', 2, 2], ['h', 0, 0]], [['h', 1, 2], ['h', 3, 0]], [['v', 0, 2], ['v', 2, 0]], [['v', 2, 2], ['v', 0, 0]], [['v', 0, 1], ['v', 2, 3]], [['v', 2, 1], ['v', 0, 3]]])
    
    def test_convertToActionNumbers(self):
        actionInfos = [[['h', 2, 0], ['h', 0, 2], ['h', 0, 0]]]
        rowNumber = 3
        columnNumber = 3
        self.assertEqual(convertToActionNumbers(actionInfos,rowNumber,columnNumber),[[0, 2, 6]])
        actionInfos = [[['h', 0, 0], ['h', 2, 2], ['h', 0, 2]], [['h', 2, 2], ['h', 0, 0], ['h', 2, 0]]]
        rowNumber = 3
        columnNumber = 3
        self.assertEqual(convertToActionNumbers(actionInfos,rowNumber,columnNumber),[[0, 2, 8], [0, 6, 8]])
    
    def test_symmetries(self):
        actionNumbers = [0]
        rowNumber = 3
        columnNumber = 3
        self.assertEqual(sorted(symmetries(actionNumbers,rowNumber,columnNumber)), sorted([[9], [2], [11], [12], [20], [15], [23]]))
        actionNumbers = [12]
        rowNumber = 3
        columnNumber = 3
        self.assertEqual(sorted(symmetries(actionNumbers,rowNumber,columnNumber)),sorted([[20], [15], [23], [0], [9], [2], [11]]))
        actionNumbers = [0, 12]
        rowNumber = 3
        columnNumber = 3 
        self.assertEqual(sorted(symmetries(actionNumbers,rowNumber,columnNumber)),sorted([[9, 20], [2, 15], [11, 23], [0, 12], [9, 20], [2, 15], [11, 23]]))
        actionNumbers = [3, 21]
        rowNumber = 3
        columnNumber = 3
        self.assertEqual(sorted(symmetries(actionNumbers,rowNumber,columnNumber)),sorted([[6, 13], [5, 22], [8, 14], [5, 13], [8, 21], [3, 14], [6, 22]]))
        actionNumbers = [0, 12]
        rowNumber = 2
        columnNumber = 3
        self.assertEqual(sorted(symmetries(actionNumbers,rowNumber,columnNumber)),sorted([[6, 16], [2, 9], [8, 13]]))
        actionNumbers = [3, 12]
        rowNumber = 2
        columnNumber = 3
        self.assertEqual(sorted(symmetries(actionNumbers,rowNumber,columnNumber)),sorted([[3, 16], [5, 9], [5, 13]]))
    
if __name__ == '__main__':
    unittest.main()