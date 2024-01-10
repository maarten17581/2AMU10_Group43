import random
import time
import math
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
from team43_A3.region import Region
from typing import List


class Cell(object):
    """
    A cel in a region in the sudoku board
    """
    
    def __init__(self, i, j, value):
        # Initiallizes the position and value of the cell, as well as the row, column and box region of the cell,
        # which are added later when creating the cells in the RegionSudokuBoard object
        self.i = i
        self.j = j
        self.value = value
        self.rowRegion: Region = 0
        self.colRegion: Region = 0
        self.boxRegion: Region = 0
        
        # Initiallize an array of counters for all values. Here a counter is the amount of cells that constrain
        # this cell from being that value, thus if the counter is 0, it is possible
        # ValueCount keeps track of the amount of values that are legal in this cell
        self.possibleValues = []
        self.valueCount = 0
        

