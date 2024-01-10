import copy
import io
import random
import time
import math
import sys
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
from team43_A3.region import Region
from team43_A3.cell import Cell
from typing import List
from functools import cmp_to_key


class RegionSudokuBoard(object):
    """
    A time efficient Sudoku Board using regions
    """
    
    # The dictionary storing the moves
    moveListDict = {}
    
    # The dictionary storing the hashes of canonical forms
    zorbristToCanonical = {}
    
    # The table that stores the random values used to generate the key value of the board
    keyGenerator = []
    
    # The table that stores the random values used to generate the move hash values
    moveKeyGenerator = []
    
    # The key value for a board
    key = 0
    
    # A dictionary from the key, to the permutation used for this position to get the canonical position
    perm = {}
    
    # The bitboard that was made last
    bitboard = []
    
    # The canonical form of the bitboard that was made last
    canonicalForm = []
    
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
            self.moveKeyGenerator.append([])
            for j in range(N):
                self.keyGenerator[i].append([])
                self.moveKeyGenerator[i].append([])
                for k in range(N+1):
                    self.keyGenerator[i][j].append(random.randint(0,sys.maxsize))
                    self.moveKeyGenerator[i][j].append(random.randint(0,sys.maxsize))

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
                    
    def getMoveHash(self, key, taboo_moves):
        """
        Returns the hash of the set of possible moves
        """
        possible_moves, mistake_moves = self.get_moves(taboo_moves)
        moves = possible_moves + mistake_moves
        hash = 0
        perm = self.perm[key]
        for move in moves:
            hash ^= self.moveKeyGenerator[perm[1].index(move[0].i)][perm[2].index(move[0].j)][perm[0].index(move[0].value-1)]
        return hash
    
    def getCanonicalHashAfterMove(self, move):
        """
        Returns the hash of the canonical form of this board after a move is made
        """
        boardHashAfterMove = self.newHashOfMove(move, False)
        
        # Check if we stored it already
        if boardHashAfterMove in self.zorbristToCanonical:
            return self.zorbristToCanonical[boardHashAfterMove]
        
        # Create the bitboard
        self.createBitBoard()
        
        # Add our new move to the bitboard
        self.bitboard[(move.value-1)][(move.i//self.m)][(move.i%self.m)][(move.j//self.n)][(move.j%self.n)] = 1
        
        # Create the canonical form
        perm = self.createCanonicalForm()
        
        # Compute the hash of the canonical form
        hash = 0
        for i in range(self.N):
            for j in range(self.N):
                for k in range(self.N):
                    if self.canonicalForm[k][(i//self.m)][(i%self.m)][(j//self.n)][(j%self.n)] == 1:
                        hash ^= self.keyGenerator[i][j][k]
        self.zorbristToCanonical[boardHashAfterMove] = hash
        self.perm[boardHashAfterMove] = perm
        return hash
    
    def getCanonicalHash(self):
        """
        Returns the hash of the canonical form of this board
        """
        boardHash = self.key
        
        # Check if we stored it already
        if boardHash in self.zorbristToCanonical:
            return self.zorbristToCanonical[boardHash]
        
        # Create the bitboard
        self.createBitBoard()
        
        # Create the canonical form
        perm = self.createCanonicalForm()
        
        # Compute the hash of the canonical form
        hash = 0
        for i in range(self.N):
            for j in range(self.N):
                for k in range(self.N):
                    if self.canonicalForm[k][(i//self.m)][(i%self.m)][(j//self.n)][(j%self.n)] == 1:
                        hash ^= self.keyGenerator[i][j][k]
        self.zorbristToCanonical[boardHash] = hash
        self.perm[boardHash] = perm
        return hash
    
    def createBitBoard(self):
        """
        Creates the bitboard from the current position
        """
        self.bitboard = [[[[[0]*self.n for _ in range(self.m)] for __ in range(self.m)] for ___ in range(self.n)] for ____ in range(self.N)]
        count = 0
        for cell in self.cells:
            if cell.value != self.emptyState:
                count += 1
                self.bitboard[(cell.value-1)][(cell.i//self.m)][(cell.i%self.m)][(cell.j//self.n)][(cell.j%self.n)] = 1
    
    def createCanonicalForm(self):
        """
        Creates the canonical form of the bitboard
        """
        
        # Comparitor for box columns
        def boxColCompare(x, y):
            xCount = 0
            yCount = 0
            for i in range(self.n):
                if x[i] == 1:
                    xCount += 1
                if y[i] == 1:
                    yCount += 1
            if xCount < yCount:
                return -1
            if xCount > yCount:
                return 1
            return 0
        
        # Comparitor for rows
        def rowCompare(x, y):
            xCount = 0
            yCount = 0
            for i in range(self.m):
                for j in range(self.n):
                    if x[i][j] == 1:
                        xCount += 1
                    if y[i][j] == 1:
                        yCount += 1
            if xCount < yCount:
                return -1
            if xCount > yCount:
                return 1
            xSorted = sorted(x, key=cmp_to_key(boxColCompare))
            ySorted = sorted(y, key=cmp_to_key(boxColCompare))
            for i in range(self.m):
                comp = boxColCompare(xSorted[i], ySorted[i])
                if comp != 0:
                    return comp
            return 0
        
        # Comparitor for box rows
        def boxRowCompare(x, y):
            xCount = 0
            yCount = 0
            for i in range(self.m):
                for j in range(self.m):
                    for k in range(self.n):
                        if x[i][j][k] == 1:
                            xCount += 1
                        if y[i][j][k] == 1:
                            yCount += 1
            if xCount < yCount:
                return -1
            if xCount > yCount:
                return 1
            xSorted = sorted(x, key=cmp_to_key(rowCompare))
            ySorted = sorted(y, key=cmp_to_key(rowCompare))
            for i in range(self.m):
                comp = rowCompare(xSorted[i], ySorted[i])
                if comp != 0:
                    return comp
            return 0
        
        # Comparitor for values
        def valueCompare(x, y):
            xCount = 0
            yCount = 0
            for i in range(self.n):
                for j in range(self.m):
                    for k in range(self.m):
                        for l in range(self.n):
                            if x[i][j][k][l] == 1:
                                xCount += 1
                            if y[i][j][k][l] == 1:
                                yCount += 1
            if xCount < yCount:
                return -1
            if xCount > yCount:
                return 1
            xSorted = sorted(x, key=cmp_to_key(boxRowCompare))
            ySorted = sorted(y, key=cmp_to_key(boxRowCompare))
            for i in range(self.n):
                comp = boxRowCompare(xSorted[i], ySorted[i])
                if comp != 0:
                    return comp
            return 0
        
        # Sort on values
        self.canonicalForm = sorted(self.bitboard, key=cmp_to_key(valueCompare))
        
        # Get value permutation
        permValues = sorted(range(len(self.bitboard)), key=cmp_to_key(lambda x, y: valueCompare(self.bitboard[x], self.bitboard[y])))
        
        # Sort on the box rows and keep track of the permutation
        intervals = [[0,self.n-1]]
        permBoxRow = list(range(self.n))
        for layer in self.canonicalForm:
            nextPermBoxRow = sorted(range(len(layer)), key=cmp_to_key(lambda x, y: boxRowCompare(layer[x], layer[y])))
            newIntervals = []
            for interval in intervals:
                subPerm = permBoxRow[interval[0]:interval[1]+1]
                sortedSubPerm = sorted(subPerm, key=lambda x: nextPermBoxRow.index(x))
                for j in range(interval[0], interval[1]+1):
                    permBoxRow[j] = sortedSubPerm[j-interval[0]]
                possibleInterval = []
                for j in range(interval[0], interval[1]+1):
                    if len(possibleInterval) == 0:
                        possibleInterval.append(j)
                    if boxRowCompare(layer[possibleInterval[-1]], layer[j]) == 0:
                        if len(possibleInterval) == 1:
                            possibleInterval.append(j)
                        else:
                            possibleInterval[1] = j
                    else:
                        if len(possibleInterval) == 2:
                            newIntervals.append(possibleInterval)
                        possibleInterval = []
                if len(possibleInterval) == 2:
                    newIntervals.append(possibleInterval)
            intervals = newIntervals
            if len(intervals) == 0:
                break
        for interval in intervals:
            sorted_perm = sorted(permBoxRow[interval[0]:interval[1]+1])
            for i in range(interval[0], interval[1]+1):
                permBoxRow[i] = sorted_perm[i-interval[0]]
        
        # Update the canonical form on this
        for i in range(len(self.canonicalForm)):
            self.canonicalForm[i] = [self.canonicalForm[i][j] for j in permBoxRow]
        
        # Sort on the individual rows within boxes, and keep track of the permutation
        rowPerms = []
        for boxRowIndex in range(self.n):
            intervals = [[0,self.m-1]]
            permRow = list(range(self.m))
            for layer in self.canonicalForm:
                boxRow = layer[boxRowIndex]
                nextPermRow = sorted(range(len(boxRow)), key=cmp_to_key(lambda x, y: rowCompare(boxRow[x], boxRow[y])))
                newIntervals = []
                for interval in intervals:
                    subPerm = permRow[interval[0]:interval[1]+1]
                    sortedSubPerm = sorted(subPerm, key=lambda x: nextPermRow.index(x))
                    for j in range(interval[0], interval[1]+1):
                        permRow[j] = sortedSubPerm[j-interval[0]]
                    possibleInterval = []
                    for j in range(interval[0], interval[1]+1):
                        if len(possibleInterval) == 0:
                            possibleInterval.append(j)
                        if rowCompare(boxRow[possibleInterval[-1]], boxRow[j]) == 0:
                            if len(possibleInterval) == 1:
                                possibleInterval.append(j)
                            else:
                                possibleInterval[1] = j
                        else:
                            if len(possibleInterval) == 2:
                                newIntervals.append(possibleInterval)
                            possibleInterval = []
                    if len(possibleInterval) == 2:
                        newIntervals.append(possibleInterval)
                intervals = newIntervals
                if len(intervals) == 0:
                    break
            for interval in intervals:
                sorted_perm = sorted(permRow[interval[0]:interval[1]+1])
                for i in range(interval[0], interval[1]+1):
                    permRow[i] = sorted_perm[i-interval[0]]
            rowPerms.append(permRow)
        
        # Update the canonical form on this
        for i in range(len(self.canonicalForm)):
            for j in range(len(self.canonicalForm[i])):
                self.canonicalForm[i][j] = [self.canonicalForm[i][j][k] for k in rowPerms[j]]
        
        # Sort on the box columns and keep track of the permutation
        intervals = [[0,self.m-1]]
        permBoxCol = list(range(self.m))
        for layer in self.canonicalForm:
            for boxRow in layer:
                for row in boxRow:
                    nextPermBoxCol = sorted(range(len(row)), key=cmp_to_key(lambda x, y: boxColCompare(row[x], row[y])))
                    newIntervals = []
                    for interval in intervals:
                        subPerm = permBoxCol[interval[0]:interval[1]+1]
                        sortedSubPerm = sorted(subPerm, key=lambda x: nextPermBoxCol.index(x))
                        for j in range(interval[0], interval[1]+1):
                            permBoxCol[j] = sortedSubPerm[j-interval[0]]
                        possibleInterval = []
                        for j in range(interval[0], interval[1]+1):
                            if len(possibleInterval) == 0:
                                possibleInterval.append(j)
                            if boxColCompare(row[possibleInterval[-1]], row[j]) == 0:
                                if len(possibleInterval) == 1:
                                    possibleInterval.append(j)
                                else:
                                    possibleInterval[1] = j
                            else:
                                if len(possibleInterval) == 2:
                                    newIntervals.append(possibleInterval)
                                possibleInterval = []
                        if len(possibleInterval) == 2:
                            newIntervals.append(possibleInterval)
                    intervals = newIntervals
                    if len(intervals) == 0:
                        break
                if len(intervals) == 0:
                        break
            if len(intervals) == 0:
                        break
        for interval in intervals:
            sorted_perm = sorted(permBoxCol[interval[0]:interval[1]+1])
            for i in range(interval[0], interval[1]+1):
                permBoxCol[i] = sorted_perm[i-interval[0]]
        
        # Update the canonical form on this
        for i in range(len(self.canonicalForm)):
            for j in range(len(self.canonicalForm[i])):
                for k in range(len(self.canonicalForm[i][j])):
                    self.canonicalForm[i][j][k] = [self.canonicalForm[i][j][k][l] for l in permBoxCol]
                   
        # Sort on the individual colums within boxes, and keep track of the permutation
        colPerms = []
        for boxColIndex in range(self.m):
            intervals = [[0,self.n-1]]
            permCol = list(range(self.n))
            for layer in self.canonicalForm:
                for boxRow in layer:
                    for row in boxRow:
                        boxCol = row[boxColIndex]
                        nextPermCol = sorted(range(len(boxCol)), key=cmp_to_key(lambda x, y: boxCol[x]-boxCol[y]))
                        newIntervals = []
                        for interval in intervals:
                            subPerm = permCol[interval[0]:interval[1]+1]
                            sortedSubPerm = sorted(subPerm, key=lambda x: nextPermCol.index(x))
                            for j in range(interval[0], interval[1]+1):
                                permCol[j] = sortedSubPerm[j-interval[0]]
                            possibleInterval = []
                            for j in range(interval[0], interval[1]+1):
                                if len(possibleInterval) == 0:
                                    possibleInterval.append(j)
                                if boxCol[possibleInterval[-1]] == boxCol[j]:
                                    if len(possibleInterval) == 1:
                                        possibleInterval.append(j)
                                    else:
                                        possibleInterval[1] = j
                                else:
                                    if len(possibleInterval) == 2:
                                        newIntervals.append(possibleInterval)
                                    possibleInterval = []
                            if len(possibleInterval) == 2:
                                newIntervals.append(possibleInterval)
                        intervals = newIntervals
                        if len(intervals) == 0:
                            break
                    if len(intervals) == 0:
                        break
                if len(intervals) == 0:
                    break
            for interval in intervals:
                sorted_perm = sorted(permCol[interval[0]:interval[1]+1])
                for i in range(interval[0], interval[1]+1):
                    permCol[i] = sorted_perm[i-interval[0]]
            colPerms.append(permCol)
        
        # Update the canonical form on this
        for i in range(len(self.canonicalForm)):
            for j in range(len(self.canonicalForm[i])):
                for k in range(len(self.canonicalForm[i][j])):
                    for l in range(len(self.canonicalForm[i][j][k])):
                        self.canonicalForm[i][j][k][l] = [self.canonicalForm[i][j][k][l][p] for p in colPerms[l]]

        # Combine the permutations for box rows and indiviual rows and for box columns and individual columns
        totalRowPermStart = list(range(self.N))
        totalColPermStart = list(range(self.N))
        totalRowPermBox = []
        totalColPermBox = []
        for i in permBoxRow:
            for j in range(self.m):
                totalRowPermBox.append(totalRowPermStart[i*self.m+j])
        for i in permBoxCol:
            for j in range(self.n):
                totalColPermBox.append(totalColPermStart[i*self.n+j])
        totalRowPerm = []
        totalColPerm = []
        for i, rowPerm in enumerate(rowPerms):
            for j in rowPerm:
                totalRowPerm.append(totalRowPermBox[i*self.m+j])
        for i, colPerm in enumerate(colPerms):
            for j in colPerm:
                totalColPerm.append(totalColPermBox[i*self.n+j])
        
        # Return the permutation used
        return (permValues, totalRowPerm, totalColPerm)
    
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
            possible_none_taboo_moves = [move for move in self.moveListDict[self.key][0] if TabooMove(move[0].i, move[0].j, move[0].value) not in tabooMoves]
            mistake_none_taboo_moves = [move for move in self.moveListDict[self.key][1] if TabooMove(move[0].i, move[0].j, move[0].value) not in tabooMoves]
            return (possible_none_taboo_moves, mistake_none_taboo_moves)
        
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
        
        none_taboo_moves = [move for move in moves if TabooMove(move[0].i, move[0].j, move[0].value) not in tabooMoves]

        # Returns the set of moves
        return none_taboo_moves, []
    
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
        self.key = self.newHashOfMove(move, False)
        
        # If the moves after this move were not computed yet, compute them and add them to the dictionary
        if not(self.key in self.moveListDict):
            self.moveListDict[self.key] = (self.getSubset(moves, move, cell), self.getSubset(mistakes, move, cell))
        if self.key in self.moveListDict:
            movesNow, mistakesNow = self.moveListDict[self.key]
            mistakesFound = [move for move in movesNow if move in mistakes]
            for mistakeMove in mistakesFound:
                movesNow.remove(mistakeMove)
                mistakesNow.append(mistakeMove)
            self.moveListDict[self.key] = (movesNow, mistakesNow)
        
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
        self.key = self.newHashOfMove(move, False)
        
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
    
    def makeRandomMove(self, taboo_moves):
        """
        Make a random move
        """
        
        # Get all possible moves we can make
        moves = [Move(cell.i, cell.j, index+1) for cell in self.cells for index, restrict in list(enumerate(cell.possibleValues)) if restrict == 0 and TabooMove(cell.i, cell.j, index+1) not in taboo_moves]
        
        # If there are none its a mistake
        if len(moves) == 0:
            return "mistake", None, None, None
        
        # Choose random move
        madeMove = random.choice(moves)
        
        # Initializing a counter for the filled regions
        regionsFilled = 0
        
        # Decrement the empty count, since an empty position is filled in
        self.emptyCount -= 1
        
        # Get the cell associated with the move
        cell = [cell for cell in self.cells if cell.i == madeMove.i and cell.j == madeMove.j][0]
        
        # Set the new value of the cell
        cell.value = madeMove.value
        
        # For the row, add the cell to the rows cell array at the proper index, increment the filled counter 
        # and if therefore the region is filled, increment the regionsFilled counter
        # Also for all other cells within the same region, remove this value as a possible value
        cell.rowRegion.filledCells[madeMove.value-1] = cell
        cell.rowRegion.filled += 1
        for otherCell in cell.rowRegion.allCells:
            if otherCell.possibleValues[madeMove.value-1] == 0:
                otherCell.valueCount -= 1
            otherCell.possibleValues[madeMove.value-1] += 1
        if cell.rowRegion.filled >= self.N:
            regionsFilled += 1
        
        # For the column, add the cell to the columns cell array at the proper index, increment the filled counter 
        # and if therefore the region is filled, increment the regionsFilled counter
        # Also for all other cells within the same region, remove this value as a possible value
        cell.colRegion.filledCells[madeMove.value-1] = cell
        cell.colRegion.filled += 1
        for otherCell in cell.colRegion.allCells:
            if otherCell.possibleValues[madeMove.value-1] == 0:
                otherCell.valueCount -= 1
            otherCell.possibleValues[madeMove.value-1] += 1
        if cell.colRegion.filled >= self.N:
            regionsFilled += 1
        
        # For the box, add the cell to the box' cell array at the proper index, increment the filled counter 
        # and if therefore the region is filled, increment the regionsFilled counter
        # Also for all other cells within the same region, remove this value as a possible value
        cell.boxRegion.filledCells[madeMove.value-1] = cell
        cell.boxRegion.filled += 1
        for otherCell in cell.boxRegion.allCells:
            if otherCell.possibleValues[madeMove.value-1] == 0:
                otherCell.valueCount -= 1
            otherCell.possibleValues[madeMove.value-1] += 1
        if cell.boxRegion.filled >= self.N:
            regionsFilled += 1
        
        self.key = self.newHashOfMove(madeMove, False)
        # Check if this move violates the sudoku rules, if so return a mistake token
        if self.mistakeMove(cell):
            self.unmakeMove(madeMove, cell)
            self.key = self.newHashOfMove(madeMove, True)
            return 0, True, madeMove, cell
        else:
            return regionsFilled, False, madeMove, cell
        
    def regionsFilledAfterMove(self, move):
        """
        Returns the amount of regions that are filled after a move is played
        """
        # Initializing a counter for the filled regions
        regionsFilled = 0
        
        # Get the cell associated with the move
        cell = [cell for cell in self.cells if move.i == cell.i and move.j == cell.j][0]
        
        # Increment the regions filled for any region that is one from being filled
        if cell.rowRegion.filled == self.N-1:
            regionsFilled += 1
        
        if cell.colRegion.filled == self.N-1:
            regionsFilled += 1
        
        if cell.boxRegion.filled == self.N-1:
            regionsFilled += 1
        
        return regionsFilled
        
    def makeMistake(self, move):
        """
        Update the stored moves to make this move a mistake move
        """
        possible_moves, mistake_moves = self.get_moves([])
        possible_moves.remove(move)
        mistake_moves.append(move)
        self.moveListDict[self.key] = (possible_moves, mistake_moves)
    
    def newHashOfMove(self, move, mistake):
        """
        Returns the new hash of the board after a move is played
        """
        if mistake:
            return self.key ^ self.keyGenerator[move.i][move.j][move.value]
        else:
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