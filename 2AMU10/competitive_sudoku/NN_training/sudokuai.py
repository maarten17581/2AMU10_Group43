#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import random
import time
import math
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai
from NN_training.regionsudokuboard import RegionSudokuBoard
from typing import List


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration.
    """
    
    # fixed point score for region filling
    pointScore = [0,1,3,7]

    def __init__(self):
        super().__init__()

    def compute_best_move(self, game_state: GameState) -> None:
        """
        The implementation of the AI Agent for Assignment 2
        @param game_state: an instance of the game provided by the simulator
        """
        
        N = game_state.board.N
        n = game_state.board.n
        m = game_state.board.m
        
        # Create our custom board data structure from the current board
        board = RegionSudokuBoard(game_state)
        
        # Create all moves for the first time using the board
        all_moves, _ = board.get_moves(game_state.taboo_moves)
        
        # Pick a random move for now
        randomMove = random.choice(all_moves)[0]
        self.propose_move(randomMove)
        
        # Start with depth 1 for iterative deepening
        depth = 1
        
        # The starting difference between the scores of the current state seeing our player as maximizing player
        dif = game_state.scores[0]-game_state.scores[1] if game_state.current_player() == 1 else game_state.scores[1]-game_state.scores[0]
        
        # Our iteritive deepening loop, if the depth is more than the number of empty squares it does not improve anymore
        while depth <= len(all_moves):
            
            # A transposition table used to check for states we have already computed
            transpositionTable = {}
            
            # Initializer for alpha and beta for the alpha beta pruning
            alpha = -math.inf
            beta = math.inf

            # Run minimax
            eval, move = self.minimax(board, game_state.taboo_moves, depth, True, dif, alpha, beta, transpositionTable)
            
            # Propose the best move for this depth and go to the next
            self.propose_move(move)
            depth += 1
    
    def minimax(self, board: RegionSudokuBoard, tabooMoves: List[TabooMove], depth: int, 
                maximizingPlayer: bool, score, alpha, beta, transpositionTable):
        """
        Runs minimax from a given position
        @param board: an instance of the given sudoku board in the {RegionSudokuBoard} datastructure
        @param tabooMoves: a list of tabooMoves given by the oracle
        @param depth: the depth left to iterate the search tree over from this state
        @param maximizingPlayer: a bool determining if the maximizing or minimizing player is at move
        @param score: the current score of the maximizing player minus the score of the minimizing player in this position
        @param alpha: the alpha parameter for alpha beta pruning
        @param beta: the beta parameter for alpha beta pruning
        @param transpositionTable: the table used to check if the position has already been computed before
        @return the score for this board state by the minimax algorithm or "mistake"
        """
        
        N = board.N
        n = board.n
        m = board.m

        # If the game is over or the depth is 0, return the evaluation of the state
        if depth == 0 or board.emptyCount == 0:
            return self.getEval(board, score, maximizingPlayer, N), None
        
        # If the board is in the transposition table, return that
        # Note here that the score and maximizingPlayer are needed to determine the uniqueness of the state for the score
        if (board.key, score, maximizingPlayer) in transpositionTable:
            return transpositionTable[(board.key, score, maximizingPlayer)]
        
        # Get the moves and mistake moves
        all_moves, lastMistakeMoves = board.get_moves(tabooMoves)
        
        # Sort the moves for alpha-beta pruning and make a copy of the mistake moves
        # to make sure previous states do not get the mistake moves calculated in this state
        sorted_moves = self.evalSort(all_moves)
        mistakeMoves = lastMistakeMoves.copy()
        
        # If there are no possible moves but the game is not over then a mistake was played,
        # return a mistake token
        if len(sorted_moves) == 0:
            return "mistake", None
        
        # Initialize a counter for possible mistake paths from here
        mistakeCount = 0
        
        # Split between maximizing or minimizing player for the minimax
        if maximizingPlayer:
            
            # Initializing the best score given maximizing player
            bestscore = -math.inf
            bestMove = None
            
            # Initialize symmetry tables
            emptyMovePlayed = []
            emptyRowAndCol = []
            emptyRow = []
            emptyCol = []
            
            # Running through all moves
            for move, cell in sorted_moves:
                
                # Search through the symmetry, if any of the symmetries have already been computed
                # we do not need to compute this move and continue
                if cell.rowRegion.filled == 0 and cell.colRegion.filled == 0 and cell.boxRegion.filled == 0:
                    if move.value in emptyMovePlayed:
                        continue
                    else:
                        emptyMovePlayed.append(move.value)
                elif cell.rowRegion.filled == 0 and cell.colRegion.filled == 0:
                    if (move.value, cell.boxRegion) in emptyRowAndCol:
                        continue
                    else:
                        emptyRowAndCol.append((move.value, cell.boxRegion))
                elif cell.rowRegion.filled == 0:
                    if (move.value, cell.colRegion, cell.boxRegion) in emptyRow:
                        continue
                    else:
                        emptyRow.append((move.value, cell.colRegion, cell.boxRegion))
                elif cell.colRegion.filled == 0:
                    if (move.value, cell.rowRegion, cell.boxRegion) in emptyCol:
                        continue
                    else:
                        emptyCol.append((move.value, cell.rowRegion, cell.boxRegion))
                
                # Make this move and update the board, it returns the amount of regions filled when making this move
                # or mistake whenever the move made was a mistake by heuristics
                regionsFilled = board.makeMove(move, cell)
                
                # If the move was a mistake do not continue and remove it from the move sets
                if regionsFilled == "mistake":
                    
                    # Add the move to the mistakes and update the stored moves within board
                    mistakeMoves.append((move, cell))
                    mistakeCount += 1
                    board.unmakeMove(move, cell)
                    newMoveset = sorted_moves.copy()
                    newMoveset.remove((move, cell))
                    board.moveListDict[board.key] = (newMoveset, mistakeMoves)
                    
                    # A mistake move wont actually be played so it cannot be in the symmetry
                    if cell.rowRegion.filled == 0 and cell.colRegion.filled == 0 and cell.boxRegion.filled == 0:
                        emptyMovePlayed.remove(move.value)
                    elif cell.rowRegion.filled == 0 and cell.colRegion.filled == 0:
                        emptyRowAndCol.remove((move.value, cell.boxRegion))
                    elif cell.rowRegion.filled == 0:
                        emptyRow.remove((move.value, cell.colRegion, cell.boxRegion))
                    elif cell.colRegion.filled == 0:
                        emptyCol.remove((move.value, cell.rowRegion, cell.boxRegion))
                    continue
                
                # Run the minimax as maximizing player
                eval, _ = self.minimax(board, tabooMoves, depth-1, False, score+self.pointScore[regionsFilled],
                                       alpha, beta, transpositionTable)

                # If this move was a mistake we add it to the mistake move set
                if eval == "mistake":
                    
                    # Add the move to the mistakes and update the stored moves within board
                    mistakeMoves.append((move, cell))
                    mistakeCount += 1
                    board.unmakeMove(move, cell)
                    newMoveset = sorted_moves.copy()
                    newMoveset.remove((move, cell))
                    board.moveListDict[board.key] = (newMoveset, mistakeMoves)
                    
                    # A mistake move wont actually be played so it cannot be in the symmetry
                    if cell.rowRegion.filled == 0 and cell.colRegion.filled == 0 and cell.boxRegion.filled == 0:
                        emptyMovePlayed.remove(move.value)
                    elif cell.rowRegion.filled == 0 and cell.colRegion.filled == 0:
                        emptyRowAndCol.remove((move.value, cell.boxRegion))
                    elif cell.rowRegion.filled == 0:
                        emptyRow.remove((move.value, cell.colRegion, cell.boxRegion))
                    elif cell.colRegion.filled == 0:
                        emptyCol.remove((move.value, cell.rowRegion, cell.boxRegion))
                    continue
                
                # If no mistake was made update the best score and move if needed and update alpha for the alpha beta pruning
                if bestscore < eval:
                    bestscore = eval
                    bestMove = move
                alpha = max(alpha, eval)
                
                # Reset the game state
                board.unmakeMove(move, cell)
                
                # If pruning is possible break out the loop and prune
                if beta <= alpha:
                    break
            
            # If all moves were mistakes then this move was a mistake, therefore we return a mistake token
            if mistakeCount >= len(sorted_moves):
                return "mistake", None
            
            # Try the mistake moves that we computed from this position
            for move, cell in mistakeMoves:
                
                # Remove the move from the set of mistake moves, since it will be a taboo move in the real game
                newMistakeMoves = mistakeMoves.copy()
                newMistakeMoves.remove((move, cell))
                
                # Make a new key for the board, since the taboo moves have changed and add the new movesets to it
                prev_moves = board.get_moves(tabooMoves)[0].copy()
                board.key ^= board.keyGenerator[move.i][move.j][move.value-1]
                board.moveListDict[board.key] = (prev_moves, newMistakeMoves)
                
                # Run minimax from the others turn on the current state
                eval, _ = self.minimax(board, tabooMoves, depth-1, False, score,
                                       alpha, beta, transpositionTable)
                
                # Reset the key of the board
                board.key ^= board.keyGenerator[move.i][move.j][move.value-1]
                
                # If this move is also a mistake, then the current move is a mistake
                if eval == "mistake":
                    return "mistake", None
                
                # If no mistake was made update the best score and move if needed and update alpha for the alpha beta pruning
                if bestscore < eval:
                    bestscore = eval
                    bestMove = move
                alpha = max(alpha, eval)
                
                # If pruning is possible break out the loop and prune
                if beta <= alpha:
                    break
        else:
            
            # Initializing the best score given minimizing player
            bestscore = math.inf
            bestMove = None
            
            # Initialize symmetry tables
            emptyMovePlayed = []
            emptyRowAndCol = []
            emptyRow = []
            emptyCol = []
            
            # Running through all moves
            for move, cell in sorted_moves:
                
                # Search through the symmetry, if any of the symmetries have already been computed
                # we do not need to compute this move and continue
                if cell.rowRegion.filled == 0 and cell.colRegion.filled == 0 and cell.boxRegion.filled == 0:
                    if move.value in emptyMovePlayed:
                        continue
                    else:
                        emptyMovePlayed.append(move.value)
                elif cell.rowRegion.filled == 0 and cell.colRegion.filled == 0:
                    if (move.value, cell.boxRegion) in emptyRowAndCol:
                        continue
                    else:
                        emptyRowAndCol.append((move.value, cell.boxRegion))
                elif cell.rowRegion.filled == 0:
                    if (move.value, cell.colRegion, cell.boxRegion) in emptyRow:
                        continue
                    else:
                        emptyRow.append((move.value, cell.colRegion, cell.boxRegion))
                elif cell.colRegion.filled == 0:
                    if (move.value, cell.rowRegion, cell.boxRegion) in emptyCol:
                        continue
                    else:
                        emptyCol.append((move.value, cell.rowRegion, cell.boxRegion))
                
                # Make this move and update the board, it returns the amount of regions filled when making this move
                # or mistake whenever the move made was a mistake by heuristics
                regionsFilled = board.makeMove(move, cell)
                
                # If the move was a mistake do not continue and remove it from the move sets
                if regionsFilled == "mistake":
                    
                    # Add the move to the mistakes and update the stored moves within board
                    mistakeMoves.append((move, cell))
                    mistakeCount += 1
                    board.unmakeMove(move, cell)
                    newMoveset = sorted_moves.copy()
                    newMoveset.remove((move, cell))
                    board.moveListDict[board.key] = (newMoveset, mistakeMoves)
                    
                    # A mistake move wont actually be played so it cannot be in the symmetry
                    if cell.rowRegion.filled == 0 and cell.colRegion.filled == 0 and cell.boxRegion.filled == 0:
                        emptyMovePlayed.remove(move.value)
                    elif cell.rowRegion.filled == 0 and cell.colRegion.filled == 0:
                        emptyRowAndCol.remove((move.value, cell.boxRegion))
                    elif cell.rowRegion.filled == 0:
                        emptyRow.remove((move.value, cell.colRegion, cell.boxRegion))
                    elif cell.colRegion.filled == 0:
                        emptyCol.remove((move.value, cell.rowRegion, cell.boxRegion))
                    continue
                
                # Run the minimax as minimizing player
                eval, _ = self.minimax(board, tabooMoves, depth-1, True, score-self.pointScore[regionsFilled],
                                       alpha, beta, transpositionTable)

                # If this move was a mistake we add it to the mistake move set
                if eval == "mistake":
                    
                    # Add the move to the mistakes and update the stored moves within board
                    mistakeMoves.append((move, cell))
                    mistakeCount += 1
                    board.unmakeMove(move, cell)
                    newMoveset = sorted_moves.copy()
                    newMoveset.remove((move, cell))
                    board.moveListDict[board.key] = (newMoveset, mistakeMoves)
                    
                    # A mistake move wont actually be played so it cannot be in the symmetry
                    if cell.rowRegion.filled == 0 and cell.colRegion.filled == 0 and cell.boxRegion.filled == 0:
                        emptyMovePlayed.remove(move.value)
                    elif cell.rowRegion.filled == 0 and cell.colRegion.filled == 0:
                        emptyRowAndCol.remove((move.value, cell.boxRegion))
                    elif cell.rowRegion.filled == 0:
                        emptyRow.remove((move.value, cell.colRegion, cell.boxRegion))
                    elif cell.colRegion.filled == 0:
                        emptyCol.remove((move.value, cell.rowRegion, cell.boxRegion))
                    continue
                
                # If no mistake was made update the best score and move if needed and update alpha for the alpha beta pruning
                if bestscore > eval:
                    bestscore = eval
                    bestMove = move
                beta = min(beta, eval)
                
                # Reset the game state
                board.unmakeMove(move, cell)
                
                # If pruning is possible break out the loop and prune
                if beta <= alpha:
                    break
            
            # If all moves were mistakes then this move was a mistake, therefore we return a mistake token
            if mistakeCount >= len(sorted_moves):
                return "mistake", None
            
            # Try the mistake moves that we computed from this position
            for move, cell in mistakeMoves:

                # Remove the move from the set of mistake moves, since it will be a taboo move in the real game
                newMistakeMoves = mistakeMoves.copy()
                newMistakeMoves.remove((move, cell))
                
                # Make a new key for the board, since the taboo moves have changed and add the new movesets to it
                prev_moves = board.get_moves(tabooMoves)[0].copy()
                board.key ^= board.keyGenerator[move.i][move.j][move.value-1]
                board.moveListDict[board.key] = (prev_moves, newMistakeMoves)
                
                # Run minimax from the others turn on the current state
                eval, _ = self.minimax(board, tabooMoves, depth-1, True, score,
                                       alpha, beta, transpositionTable)
                
                # Reset the key of the board
                board.key ^= board.keyGenerator[move.i][move.j][move.value-1]
                
                # If this move is also a mistake, then the current move is a mistake
                if eval == "mistake":
                    return "mistake", None
                
                # If no mistake was made update the best score and move if needed and update alpha for the alpha beta pruning
                if bestscore > eval:
                    bestscore = eval
                    bestMove = move
                beta = min(beta, eval)
                
                # If pruning is possible break out the loop and prune
                if beta <= alpha:
                    break
        
        # Add the best score and best move for this state to the transposition table
        transpositionTable[(board.key, score, maximizingPlayer)] = (bestscore, bestMove)
        
        # If no mistake was detected return the best score and move
        return bestscore, bestMove

    def getEval(self, board, score, maximizingPlayer, N):
        """
        Returns the evaluation function of a given state
        @param board: the board state of the game
        @param score: the difference in scores of both players
        @param maximizingPlayer: a boolean that determines if the maximizing player is at move
        @return the evaluation of the state
        """
        
        # Reward the AI for empty regions and punish the AI for regions that are filled untill one cell is left
        regionSum = 0
        for rowRegion in board.rowRegions:
            if rowRegion.filled == 0:
                regionSum += 1
            if rowRegion.filled == N-1:
                regionSum -= N
        for colRegion in board.colRegions:
            if colRegion.filled == 0:
                regionSum += 1
            if colRegion.filled == N-1:
                regionSum -= N
        for boxRegion in board.boxRegions:
            if boxRegion.filled == 0:
                regionSum += 1
            if boxRegion.filled == N-1:
                regionSum -= N
                
        # Add the score, the regionSum and a value based on whether we are the last playing player or not together
        # Note here that as regionSum < 3*N any point scored is better then any amount of regionSum or last playing player bonus
        # as well as the last playing player bonus always being better then any amount of regionSum
        return score*8*N + regionSum + (4*N if (maximizingPlayer and board.emptyCount % 2 == 1) \
                                        or (not maximizingPlayer and board.emptyCount % 2 != 1) else -4*N)
    
    def evalSort(self, moves):
        """
        Sorts the moves in an order to increase pruning from alpha-beta pruning
        @param moves: the moves to be sorted
        @return the list of moves, sorted based on an order
        """
        
        # The moves are sorted by the amount of values in the regions that they are part of in descending order
        # This increases likelyhood of regions staying empty and increases searching in more restricted cells
        sortingOrder = []
        for move, cell in moves:
            sortingOrder.append(cell.rowRegion.filled + cell.colRegion.filled + cell.boxRegion.filled)
        
        return [move for _, move in sorted(zip(sortingOrder, moves), key=lambda pair: -pair[0])]
