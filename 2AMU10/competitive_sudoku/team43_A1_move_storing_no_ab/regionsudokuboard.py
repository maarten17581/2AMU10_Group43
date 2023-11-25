import random
import time
import math
import sys
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
from team43_A1_move_storing_no_ab.region import Region
from team43_A1_move_storing_no_ab.cell import Cell
from typing import List


class RegionSudokuBoard(object):
    """
    A time efficient Sudoku Board
    """
    
    keyGenerator = []
    
    key = 0
    
    emptyState = 0
    
    emptyCount = 0
    
    cells = []
    
    rowRegions = []
    colRegions = []
    boxRegions = []
    n = 0
    m = 0
    N = 0
    
    def __init__(self, game_state: GameState):
        n = game_state.board.n
        m = game_state.board.m
        N = game_state.board.N
        self.n = n
        self.m = m
        self.N = N
        """
        Create hash key table
        """
        for i in range(N):
            self.keyGenerator.append([])
            for j in range(N):
                self.keyGenerator[i].append([])
                for k in range(N+1):
                    self.keyGenerator[i][j].append(random.randint(0,sys.maxsize))
        """
        Create the board given the gamestate
        """
        self.emptyState = game_state.board.empty
        for i in range(N):
            newRowRegion = Region(n, m, N, self.emptyState)
            newColRegion = Region(n, m, N, self.emptyState)
            newBoxRegion = Region(n, m, N, self.emptyState)
            self.rowRegions.append(newRowRegion)
            self.colRegions.append(newColRegion)
            self.boxRegions.append(newBoxRegion)
        for i in range(N):
            for j in range(N):
                value = game_state.board.get(i, j)
                self.key ^= self.keyGenerator[i][j][value]
                cell = Cell(i, j, value)
                row = i
                col = j
                box = (i//m)*m + j//n
                cell.rowRegion = self.rowRegions[row]
                cell.colRegion = self.colRegions[col]
                cell.boxRegion = self.boxRegions[box]
                self.cells.append(cell)
                if value == self.emptyState:
                    self.emptyCount += 1
                else:
                    cell.rowRegion.filled += 1
                    cell.colRegion.filled += 1
                    cell.boxRegion.filled += 1
                    cell.rowRegion.cells[value-1] = cell
                    cell.colRegion.cells[value-1] = cell
                    cell.boxRegion.cells[value-1] = cell
                    
    
    def get_moves(self, tabooMoves: List[TabooMove]):
        moves = []
        for cell in self.cells:
            if cell.value != self.emptyState:
                continue
            i = cell.i
            j = cell.j
            for value in range(1, self.N+1):
                if TabooMove(i, j, value) in tabooMoves:
                    continue
                if cell.rowRegion.cells[value-1] != self.emptyState or \
                    cell.colRegion.cells[value-1] != self.emptyState or \
                    cell.boxRegion.cells[value-1] != self.emptyState:
                    continue
                moves.append((Move(i, j, value), cell))
        return moves
    
    def makeMove(self, move: Move, cell: Cell):
        self.key ^= self.keyGenerator[move.i][move.j][0] ^ self.keyGenerator[move.i][move.j][move.value]
        regionsFilled = 0
        self.emptyCount -= 1
        cell.value = move.value
        cell.rowRegion.cells[move.value-1] = cell
        cell.rowRegion.filled += 1
        if cell.rowRegion.filled >= self.N:
            regionsFilled += 1
        cell.colRegion.cells[move.value-1] = cell
        cell.colRegion.filled += 1
        if cell.colRegion.filled >= self.N:
            regionsFilled += 1
        cell.boxRegion.cells[move.value-1] = cell
        cell.boxRegion.filled += 1
        if cell.boxRegion.filled >= self.N:
            regionsFilled += 1
        return regionsFilled
    
    def unmakeMove(self, move: Move, cell: Cell):
        self.key ^= self.keyGenerator[move.i][move.j][0] ^ self.keyGenerator[move.i][move.j][move.value]
        self.emptyCount += 1
        cell.rowRegion.cells[cell.value-1] = self.emptyState
        cell.rowRegion.filled -= 1
        cell.colRegion.cells[cell.value-1] = self.emptyState
        cell.colRegion.filled -= 1
        cell.boxRegion.cells[cell.value-1] = self.emptyState
        cell.boxRegion.filled -= 1
        cell.value = self.emptyState

