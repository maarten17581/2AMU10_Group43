import random
import time
import math
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
from typing import List


class Region(object):
    """
    A region in the sudoku board
    """
    
    def __init__(self, n, m, N, emptyState):
        self.n = n
        self.m = m
        self.N = N
        
        # Initialize the array of cells that are contained within this region.
        # They are sorted by value, where the index corresponds to the value of the cell.
        # This is done for efficient value finding
        self.filledCells = []
        
        self.allCells = []
        
        # A counter for the number of values that are already placed in this region
        self.filled = 0
        
        # Initializes the cells as empty spaces
        for i in range(N):
            self.filledCells.append(emptyState)