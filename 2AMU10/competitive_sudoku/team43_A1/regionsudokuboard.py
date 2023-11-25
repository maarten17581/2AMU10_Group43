import random
import time
import math
import sys
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
from team43_A1.region import Region
from team43_A1.cell import Cell
from typing import List


class RegionSudokuBoard(object):
    """
    A time efficient Sudoku Board using regions
    """
    
    # The table that stores the random values used to generate the key value of the board
    keyGenerator = []
    
    # The key value for a board
    key = 0
    
    # The value used for the empty cell
    emptyState = 0
    
    # The amount of empty squares in the sudoku
    emptyCount = 0
    
    # An array of all Cell objects in the sudoku
    cells = []
    
    # Arrays for all rows, columns and boxes where we have one Region object for each one of these
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
        
        # Create key table
        for i in range(N):
            self.keyGenerator.append([])
            for j in range(N):
                self.keyGenerator[i].append([])
                for k in range(N+1):
                    self.keyGenerator[i][j].append(random.randint(0,sys.maxsize))

        # Initiallize the region objects and the empty state
        self.emptyState = game_state.board.empty
        for i in range(N):
            newRowRegion = Region(n, m, N, self.emptyState)
            newColRegion = Region(n, m, N, self.emptyState)
            newBoxRegion = Region(n, m, N, self.emptyState)
            self.rowRegions.append(newRowRegion)
            self.colRegions.append(newColRegion)
            self.boxRegions.append(newBoxRegion)
           
        # For every position in the sudoku do the following
        for i in range(N):
            for j in range(N):
                
                # Get the value that this cell has in the sudoku
                value = game_state.board.get(i, j)
                
                # Add the key value to the key using a bitwise XOR to update the key
                self.key ^= self.keyGenerator[i][j][value]
                
                # Make a Cell object and add the right row, column and box region to the cell which contain this cell
                cell = Cell(i, j, value)
                row = i
                col = j
                box = (i//m)*m + j//n
                cell.rowRegion = self.rowRegions[row]
                cell.colRegion = self.colRegions[col]
                cell.boxRegion = self.boxRegions[box]
                
                # Add the cell to the set of all cells
                self.cells.append(cell)
                
                # If this cell is empty increment the empty counter
                if value == self.emptyState:
                    self.emptyCount += 1
                else:
                    # Else we increment the filled counter for each region that contains the cell
                    # and add the cell in the regions cells array at the proper index
                    cell.rowRegion.filled += 1
                    cell.colRegion.filled += 1
                    cell.boxRegion.filled += 1
                    cell.rowRegion.cells[value-1] = cell
                    cell.colRegion.cells[value-1] = cell
                    cell.boxRegion.cells[value-1] = cell
                    
    
    def get_moves(self, tabooMoves: List[TabooMove]):
        """
        Generates all moves for this board
        @param tabooMoves: A list of taboo moves that are ruled out as possible moves
        @return a set of tuples of 2 elements, the first of which is a move and the second the cell 
            corresponding to this move. There is a tuple in the set for every allowed move. The cell
            is used for the makeMove() and unmakeMove() methods so are passed back as satalite data.
        """
        
        # Initializing the set of moves
        moves = []
        
        # Run over every cell and do the following
        for cell in self.cells:
            
            # If the cell is not empty then no moves are possible, so continue with the next cell
            if cell.value != self.emptyState:
                continue
            i = cell.i
            j = cell.j
            
            # For every value do the following
            for value in range(1, self.N+1):
                
                # If this position with this value is a taboo move in the list,
                # then this is not possible and we go to the next value
                if TabooMove(i, j, value) in tabooMoves:
                    continue
                
                # If in the row, column or box that this cell is in already contains this value,
                # then this is not possible and we go to the next value
                if cell.rowRegion.cells[value-1] != self.emptyState or \
                    cell.colRegion.cells[value-1] != self.emptyState or \
                    cell.boxRegion.cells[value-1] != self.emptyState:
                    continue
                
                # If none of this is the case we add the move to the set of moves
                moves.append((Move(i, j, value), cell))

        # Returns the set of moves
        return moves
    
    def makeMove(self, move: Move, cell: Cell):
        """
        Updates the board after the move is made
        @param move: the move that is applied to the sudoku
        @param cell: the cell that corresponds to the move, as the get_moves() returns with every move
        @return the amount of regions that are filled when applying this move to the sudoku
        """
        
        # Update the key of the sudoku by adding the key value of this cell at the given value and the 0 value using bitwise XOR
        self.key ^= self.keyGenerator[move.i][move.j][0] ^ self.keyGenerator[move.i][move.j][move.value]
        
        # Initializing a counter for the filled regions
        regionsFilled = 0
        
        # Decrement the empty count, since an empty position is filled in
        self.emptyCount -= 1
        
        # Set the new value of the cell
        cell.value = move.value
        
        # For the row, add the cell to the rows cell array at the proper index, increment the filled counter 
        # and if therefore the region is filled, increment the regionsFilled counter
        cell.rowRegion.cells[move.value-1] = cell
        cell.rowRegion.filled += 1
        if cell.rowRegion.filled >= self.N:
            regionsFilled += 1
        
        # For the column, add the cell to the columns cell array at the proper index, increment the filled counter 
        # and if therefore the region is filled, increment the regionsFilled counter
        cell.colRegion.cells[move.value-1] = cell
        cell.colRegion.filled += 1
        if cell.colRegion.filled >= self.N:
            regionsFilled += 1
        
        # For the box, add the cell to the box' cell array at the proper index, increment the filled counter 
        # and if therefore the region is filled, increment the regionsFilled counter
        cell.boxRegion.cells[move.value-1] = cell
        cell.boxRegion.filled += 1
        if cell.boxRegion.filled >= self.N:
            regionsFilled += 1
        
        # Return the amount of regions that are filled
        return regionsFilled
    
    def unmakeMove(self, move: Move, cell: Cell):
        """
        Updates the board to the state before the move is made
        @param move: the move that was applied to the sudoku
        @param cell: the cell that corresponds to the move, as the get_moves() returns with every move
        """
        
        # Update the key of the sudoku by adding the key value of this cell at the given value and the 0 value using bitwise XOR
        self.key ^= self.keyGenerator[move.i][move.j][0] ^ self.keyGenerator[move.i][move.j][move.value]
        
        # Increment the empty count, since an empty position is filled in
        self.emptyCount += 1
        
        # For the row, remove the cell from the rows cell array at the proper index and decrement the filled counter
        cell.rowRegion.cells[cell.value-1] = self.emptyState
        cell.rowRegion.filled -= 1
        
        # For the column, remove the cell from the columns cell array at the proper index and decrement the filled counter
        cell.colRegion.cells[cell.value-1] = self.emptyState
        cell.colRegion.filled -= 1
        
        # For the box, remove the cell from the box' cell array at the proper index and decrement the filled counter
        cell.boxRegion.cells[cell.value-1] = self.emptyState
        cell.boxRegion.filled -= 1
        
        # Reset the value of the cell
        cell.value = self.emptyState

