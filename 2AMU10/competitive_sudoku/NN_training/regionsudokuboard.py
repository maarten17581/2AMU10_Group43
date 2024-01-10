import random
import time
import math
import sys
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
from NN_training.region import Region
from NN_training.cell import Cell
from typing import List
import io
import numpy as np


class RegionSudokuBoard(object):
    """
    A time efficient Sudoku Board using regions
    """
    
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
        
        # The dictionary storing the moves
        self.moveListDict = {}
        
        # The table that stores the random values used to generate the key value of the board
        self.keyGenerator = []
        
        # The key value for a board
        self.key = 0
        
        # The value used for the empty cell
        self.emptyState = 0
        
        # The amount of empty squares in the sudoku
        self.emptyCount = 0
        
        # An array of all Cell objects in the sudoku
        self.cells = []
        
        # Arrays for all rows, columns and boxes where we have one Region object for each one of these
        self.rowRegions = []
        self.colRegions = []
        self.boxRegions = []
        
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
                
                # Add the cell within the region also to the region
                self.rowRegions[row].allCells.append(cell)
                self.colRegions[col].allCells.append(cell)
                self.boxRegions[box].allCells.append(cell)
                
                for _ in range(N):
                    cell.possibleValues.append(0)
                
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
                    cell.rowRegion.filledCells[value-1] = cell
                    cell.colRegion.filledCells[value-1] = cell
                    cell.boxRegion.filledCells[value-1] = cell
        
        # Updating the possible values that cells can have
        for cell in self.cells:
            for value in range(N):
                if cell.rowRegion.filledCells[value] != self.emptyState:
                    cell.possibleValues[value] += 1
                if cell.colRegion.filledCells[value] != self.emptyState:
                    cell.possibleValues[value] += 1
                if cell.boxRegion.filledCells[value] != self.emptyState:
                    cell.possibleValues[value] += 1
                if cell.possibleValues[value] == 0:
                    cell.valueCount += 1
                    
    
    def get_moves(self, tabooMoves: List[TabooMove]):
        """
        Generates all moves for this board
        @param tabooMoves: A list of taboo moves that are ruled out as possible moves
        @return a set of tuples of 2 elements, the first of which is a move and the second the cell 
            corresponding to this move. There is a tuple in the set for every allowed move. The cell
            is used for the makeMove() and unmakeMove() methods so are passed back as satalite data.
        """
        
        # If these moves have already been computed before, return that list
        if self.key in self.moveListDict:
            return self.moveListDict[self.key]
        
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
                if cell.rowRegion.filledCells[value-1] != self.emptyState or \
                    cell.colRegion.filledCells[value-1] != self.emptyState or \
                    cell.boxRegion.filledCells[value-1] != self.emptyState:
                    continue
                
                # If none of this is the case we add the move to the set of moves
                moves.append((Move(i, j, value), cell))
        
        # After computing all these moves, add it to the table for whenever it is requested again
        self.moveListDict[self.key] = (moves, [])

        # Returns the set of moves
        return moves, []
    
    def makeMove(self, move: Move, cell: Cell):
        """
        Updates the board after the move is made
        @param move: the move that is applied to the sudoku
        @param cell: the cell that corresponds to the move, as the get_moves() returns with every move
        @return the amount of regions that are filled when applying this move to the sudoku
        """
        
        # Get the moves of the current state before the move
        moves, mistakes = self.moveListDict[self.key]
        
        # Update the key of the sudoku used for retrieving the moves
        self.key = self.newHashOfMove(move)
        
        # If the moves after this move were not computed yet, compute them and add them to the dictionary
        if not(self.key in self.moveListDict):
            self.moveListDict[self.key] = (self.getSubset(moves, move, cell), self.getSubset(mistakes, move, cell))
        
        # Initializing a counter for the filled regions
        regionsFilled = 0
        
        # Decrement the empty count, since an empty position is filled in
        self.emptyCount -= 1
        
        # Set the new value of the cell
        cell.value = move.value
        
        # For the row, add the cell to the rows cell array at the proper index, increment the filled counter 
        # and if therefore the region is filled, increment the regionsFilled counter
        # Also for all other cells within the same region, remove this value as a possible value
        cell.rowRegion.filledCells[move.value-1] = cell
        cell.rowRegion.filled += 1
        for otherCell in cell.rowRegion.allCells:
            if otherCell.possibleValues[move.value-1] == 0:
                otherCell.valueCount -= 1
            otherCell.possibleValues[move.value-1] += 1
        if cell.rowRegion.filled >= self.N:
            regionsFilled += 1
        
        # For the column, add the cell to the columns cell array at the proper index, increment the filled counter 
        # and if therefore the region is filled, increment the regionsFilled counter
        # Also for all other cells within the same region, remove this value as a possible value
        cell.colRegion.filledCells[move.value-1] = cell
        cell.colRegion.filled += 1
        for otherCell in cell.colRegion.allCells:
            if otherCell.possibleValues[move.value-1] == 0:
                otherCell.valueCount -= 1
            otherCell.possibleValues[move.value-1] += 1
        if cell.colRegion.filled >= self.N:
            regionsFilled += 1
        
        # For the box, add the cell to the box' cell array at the proper index, increment the filled counter 
        # and if therefore the region is filled, increment the regionsFilled counter
        # Also for all other cells within the same region, remove this value as a possible value
        cell.boxRegion.filledCells[move.value-1] = cell
        cell.boxRegion.filled += 1
        for otherCell in cell.boxRegion.allCells:
            if otherCell.possibleValues[move.value-1] == 0:
                otherCell.valueCount -= 1
            otherCell.possibleValues[move.value-1] += 1
        if cell.boxRegion.filled >= self.N:
            regionsFilled += 1
        
        # Check if this move violates the sudoku rules, if so return a mistake token
        if self.mistakeMove(cell):
            return "mistake"
        
        # Return the amount of regions that are filled
        return regionsFilled
    
    def unmakeMove(self, move: Move, cell: Cell):
        """
        Updates the board to the state before the move is made
        @param move: the move that was applied to the sudoku
        @param cell: the cell that corresponds to the move, as the get_moves() returns with every move
        """
        
        # Update the key of the sudoku by adding the key value of this cell at the given value and the 0 value using bitwise XOR
        self.key = self.newHashOfMove(move)
        
        # Increment the empty count, since an empty position is filled in
        self.emptyCount += 1
        
        # For the row, remove the cell from the rows cell array at the proper index and decrement the filled counter
        # Also for all other cells within the same region, add this value as a possible value if no other cells contrain it
        cell.rowRegion.filledCells[cell.value-1] = self.emptyState
        cell.rowRegion.filled -= 1
        for otherCell in cell.rowRegion.allCells:
            otherCell.possibleValues[move.value-1] -= 1
            if otherCell.possibleValues[move.value-1] == 0:
                otherCell.valueCount += 1
        
        # For the column, remove the cell from the columns cell array at the proper index and decrement the filled counter
        # Also for all other cells within the same region, add this value as a possible value if no other cells contrain it
        cell.colRegion.filledCells[cell.value-1] = self.emptyState
        cell.colRegion.filled -= 1
        for otherCell in cell.colRegion.allCells:
            otherCell.possibleValues[move.value-1] -= 1
            if otherCell.possibleValues[move.value-1] == 0:
                otherCell.valueCount += 1
        
        # For the box, remove the cell from the box' cell array at the proper index and decrement the filled counter
        # Also for all other cells within the same region, add this value as a possible value if no other cells contrain it
        cell.boxRegion.filledCells[cell.value-1] = self.emptyState
        cell.boxRegion.filled -= 1
        for otherCell in cell.boxRegion.allCells:
            otherCell.possibleValues[move.value-1] -= 1
            if otherCell.possibleValues[move.value-1] == 0:
                otherCell.valueCount += 1
        
        # Reset the value of the cell
        cell.value = self.emptyState
        
        
    def getSubset(self, all_moves, move, cell):
        """
        Computes all possible moves given a move was played
        @param all_moves: a set of all possible moves before the move was played
        @param move: the move that is played
        @param cell: the cell in the {RegionSudokuBoard} datastructure that the move changes the value of
        @return a subset of all_moves where all moves are removed that are in violation with move
        """
        
        # Initializing the subset
        subsetMoves = []
        
        # Check for all moves if they are in violation with move
        for posMove, posCell in all_moves:
            
            # They are in violation if the move is in the same position, 
            # or whenever it is the same value within the same row, column or box
            if (move.i == posMove.i and move.j == posMove.j) or (move.value == posMove.value and 
                    (cell.rowRegion is posCell.rowRegion or cell.colRegion is posCell.colRegion or
                    cell.boxRegion is posCell.boxRegion)):
                continue
            
            # If this is not the case add it to the subset
            subsetMoves.append((posMove, posCell))
            
        # Return the subset of moves
        return subsetMoves
    
    def mistakeMove(self, cell):
        """
        Determines if a move is illegal by the sudoku rules, and thus will be considered a mistake move
        """
        
        # For all cells in the row
        for otherCell in cell.rowRegion.allCells:
            
            # If the cell had a value, then the move does not constrain it
            if otherCell.value != self.emptyState:
                continue
            
            # If no value can be added to the cell, then the move was not allowed
            if otherCell.valueCount == 0:
                return True
            
            # If for the column that this cell is part of, no cell can have this
            # value as possible value, then the move was not allowed
            if otherCell.colRegion.filledCells[cell.value-1] == self.emptyState:
                stillPossible = False
                for regionCell in otherCell.colRegion.allCells:
                    if regionCell.value != self.emptyState:
                        continue
                    if regionCell.possibleValues[cell.value-1] == 0:
                        stillPossible = True
                        break
                if not stillPossible:
                    return True
            
            # If for the box that this cell is part of, no cell can have this
            # value as possible value, then the move was not allowed
            if otherCell.boxRegion.filledCells[cell.value-1] == self.emptyState:
                stillPossible = False
                for regionCell in otherCell.boxRegion.allCells:
                    if regionCell.value != self.emptyState:
                        continue
                    if regionCell.possibleValues[cell.value-1] == 0:
                        stillPossible = True
                        break
                if not stillPossible:
                    return True
        
        # For all cells in the column
        for otherCell in cell.colRegion.allCells:
            
            # If the cell had a value, then the move does not constrain it
            if otherCell.value != self.emptyState:
                continue
            
            # If no value can be added to the cell, then the move was not allowed
            if otherCell.valueCount == 0:
                return True
            
            # If for the row that this cell is part of, no cell can have this
            # value as possible value, then the move was not allowed
            if otherCell.rowRegion.filledCells[cell.value-1] == self.emptyState:
                stillPossible = False
                for regionCell in otherCell.rowRegion.allCells:
                    if regionCell.value != self.emptyState:
                        continue
                    if regionCell.possibleValues[cell.value-1] == 0:
                        stillPossible = True
                        break
                if not stillPossible:
                    return True
                
            # If for the box that this cell is part of, no cell can have this
            # value as possible value, then the move was not allowed
            if otherCell.boxRegion.filledCells[cell.value-1] == self.emptyState:
                stillPossible = False
                for regionCell in otherCell.boxRegion.allCells:
                    if regionCell.value != self.emptyState:
                        continue
                    if regionCell.possibleValues[cell.value-1] == 0:
                        stillPossible = True
                        break
                if not stillPossible:
                    return True
        
        # For all cells in the box
        for otherCell in cell.boxRegion.allCells:
            
            # If the cell had a value, then the move does not constrain it
            if otherCell.value != self.emptyState:
                continue
            
            # If no value can be added to the cell, then the move was not allowed
            if otherCell.valueCount == 0:
                return True
            
            # If for the row that this cell is part of, no cell can have this
            # value as possible value, then the move was not allowed
            if otherCell.rowRegion.filledCells[cell.value-1] == self.emptyState:
                stillPossible = False
                for regionCell in otherCell.rowRegion.allCells:
                    if regionCell.value != self.emptyState:
                        continue
                    if regionCell.possibleValues[cell.value-1] == 0:
                        stillPossible = True
                        break
                if not stillPossible:
                    return True
                
            # If for the column that this cell is part of, no cell can have this
            # value as possible value, then the move was not allowed
            if otherCell.colRegion.filledCells[cell.value-1] == self.emptyState:
                stillPossible = False
                for regionCell in otherCell.colRegion.allCells:
                    if regionCell.value != self.emptyState:
                        continue
                    if regionCell.possibleValues[cell.value-1] == 0:
                        stillPossible = True
                        break
                if not stillPossible:
                    return True
        
        # If none of these contraints were an issue, then this move is fine as far as we know
        return False
    
    def newHashOfMove(self, move):
        """
        Returns the new hash of the board after a move is played
        """
        return self.key ^ self.keyGenerator[move.i][move.j][0] ^ self.keyGenerator[move.i][move.j][move.value]
    
    def __str__(self) -> str:
        """
        Prints the board in a simple textual format. The first line contains the values m and n. Then the contents of
        the rows are printed as space separated lists, where a dot '.' is used to represent an empty square.
        @return: The generated string.
        """
        

        m = self.m
        n = self.n
        N = self.N
        out = io.StringIO()

        def print_square(i, j):
            value = [cell.value for cell in self.cells if cell.i == i and cell.j == j][0]
            s = '   .' if value == 0 else f'{value:>4}'
            out.write(s)

        out.write(f'{m} {n}\n')
        for i in range(N):
            for j in range(N):
                print_square(i, j)
            out.write('\n')
        return out.getvalue()