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
        
        # An array of only those cells that have a value, stored at the index of there value for efficient searching
        self.filledCells = []
        
        # An array of all cells that are part of the region
        self.allCells = []
        
        # A counter for the number of values that are already placed in this region
        self.filled = 0
        
        # Initializes the cells as empty spaces
        for i in range(N):
            self.filledCells.append(emptyState)