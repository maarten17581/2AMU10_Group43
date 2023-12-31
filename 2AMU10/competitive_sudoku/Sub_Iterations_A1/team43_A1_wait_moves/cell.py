import random
import time
import math
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
from team43_A1_wait_moves.region import Region
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
        

