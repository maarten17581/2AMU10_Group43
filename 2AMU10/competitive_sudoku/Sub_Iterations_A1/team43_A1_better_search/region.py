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
        self.cells = []
        self.filled = 0
        for i in range(N):
            self.cells.append(emptyState)