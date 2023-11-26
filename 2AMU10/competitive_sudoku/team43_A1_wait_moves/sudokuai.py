#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import random
import time
import math
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai
from team43_A1_wait_moves.regionsudokuboard import RegionSudokuBoard
from typing import List


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration.
    """
    
    # fixed point score for region filling
    pointScore = [0,1,3,7]

    def __init__(self):
        super().__init__()

    # N.B. This is a very naive implementation.
    def compute_best_move(self, game_state: GameState) -> None:
        N = game_state.board.N
        n = game_state.board.n
        m = game_state.board.m
        
        # Create our custom board data structure from the current board
        board = RegionSudokuBoard(game_state)
        
        # Create all moves for the first time using the board
        all_moves = board.get_moves(game_state.taboo_moves)
        
        # Pick a random move for now
        randomMove = random.choice(all_moves)[0]
        self.propose_move(randomMove)
        
        # Start with depth 0 for iterative deepening
        depth = 0
        
        # The starting difference between the scores of the current state seeing our player as maximizing player
        dif = game_state.scores[0]-game_state.scores[1] if game_state.current_player() == 1 else game_state.scores[1]-game_state.scores[0]
        
        # The dictionary used for storing our move sets
        moveListDict = {}
        
        # Our iteritive deepening loop, if the depth is more than the number of empty squares it does not improve anymore
        while depth <= board.emptyCount:
            
            # The best move and corresponding score that is found in this depth
            bestscoreThisDepth = -math.inf
            bestmoveThisDepth = randomMove
            
            # Initializer for alpha and beta for the alpha beta pruning
            alpha = -math.inf
            beta = math.inf
            
            # Running through all moves for our first minimax step
            for move, cell in all_moves:
                
                # Make this move and update the board, it returns the amount of regions filled when making this move
                regionsFilled = board.makeMove(move, cell)
                
                # Check if we already computed the set of moves for this state,
                # if so retrive it from the dictionary, otherwise compute it
                subsetMoves = 0
                if board.key in moveListDict:
                    subsetMoves = moveListDict.get(board.key)
                else:
                    subsetMoves = self.getSubset(all_moves, move, cell)
                    moveListDict[board.key] = subsetMoves
                
                # Run the minimax as maximizing player
                eval = self.minimax(board, game_state.taboo_moves, depth, False, 
                                    dif+self.pointScore[regionsFilled],
                                    alpha, beta, subsetMoves, moveListDict)

                # If this move was a mistake we check if waiting in this possition is better, by playing this mistaken move
                if eval == "mistake":
                    
                    # Reset the move
                    board.unmakeMove(move, cell)
                    
                    # Remove the move from the set of moves, since it will be a taboo move in the real game
                    newMoveset = all_moves.copy()
                    newMoveset.remove((move, cell))
                    
                    # Make a new key for the board, since the taboo moves have changed
                    board.key ^= board.keyGenerator[move.i][move.j][move.value-1]
                    
                    # Run minimax from the others turn on the current state,
                    # with a little worse scoring to not encourage this when having multiple options
                    eval = self.minimax(board, game_state.taboo_moves, depth, False, 
                                        dif-(1/2), alpha, beta, newMoveset, moveListDict)
                    
                    # Reset the key of the board
                    board.key ^= board.keyGenerator[move.i][move.j][move.value-1]
                    
                    # Make the move to make sure later the reset goes well
                    board.makeMove(move, cell)
                
                # If this is the new best move store it
                if eval > bestscoreThisDepth:
                    bestmoveThisDepth = move
                    bestscoreThisDepth = eval
                
                # Update the alpha beta pruning
                alpha = max(alpha, eval)
                if beta <= alpha:
                    board.unmakeMove(move, cell)
                    break
                
                # Reset the game state for the next move
                board.unmakeMove(move, cell)
            
            # Propose the best move for this depth and go to the next
            self.propose_move(bestmoveThisDepth)
            depth += 1
    
    def minimax(self, board: RegionSudokuBoard, tabooMoves: List[TabooMove], depth: int, maximizingPlayer: bool, score,
                alpha, beta, all_moves, moveListDict):
        """
        Runs minimax from a given position
        @param board: an instance of the given sudoku board in the {RegionSudokuBoard} datastructure
        @param tabooMoves: a list of tabooMoves given by the oracle
        @param depth: the depth left to iterate the search tree over from this state
        @param maximizingPlayer: a bool determining if the maximizing or minimizing player is at move
        @param score: the current score of the maximizing player minus the score of the minimizing player in this position
        @param alpha: the alpha parameter for alpha beta pruning
        @param beta: the beta parameter for alpha beta pruning
        @param all_moves: the set of all possible moves one can make from this position
        @param moveListDict: a dictionary of sets of possible moves given a position, to remove recomputation in iteritive improvement
        @return the score for this board state by the minimax algorithm
        """

        N = board.N
        n = board.n
        m = board.m

        # If the game is over or the depth is 0, return the score difference
        if depth == 0 or board.emptyCount == 0:
            return score
        
        # If there are no possible moves but the game is not over then a mistake was played,
        # return a mistake token
        if len(all_moves) == 0:
            return "mistake"
        
        # Check if any empty square does not have a valid move, if so return a mistake token
        for cell in board.cells:
            if cell.value == board.emptyState:
                noValue = True
                for value in range(N):
                    if cell.rowRegion.cells[value] == board.emptyState and cell.colRegion.cells[value] == board.emptyState and \
                            cell.boxRegion.cells[value] == board.emptyState:
                        noValue = False
                        break
                if noValue:
                    return "mistake"
        
        # Initialize a counter for possible mistake paths from here
        mistakeCount = 0
        
        # Split between maximizing or minimizing player for the minimax
        if maximizingPlayer:
            
            # Initializing the best score given maximizing player
            bestscore = -math.inf
            
            # Running through all moves
            for move, cell in all_moves:
                
                # Make this move and update the board, it returns the amount of regions filled when making this move
                regionsFilled = board.makeMove(move, cell)
                
                # Check if we already computed the set of moves for this state,
                # if so retrive it from the dictionary, otherwise compute it
                subsetMoves = 0
                if board.key in moveListDict:
                    subsetMoves = moveListDict.get(board.key)
                else:
                    subsetMoves = self.getSubset(all_moves, move, cell)
                    moveListDict[board.key] = subsetMoves
                
                # Run the minimax as minimizing player
                eval = self.minimax(board, tabooMoves, depth-1, False, score+self.pointScore[regionsFilled],
                                    alpha, beta, subsetMoves, moveListDict)

                # If this move was a mistake we check if waiting in this possition is better, by playing this mistaken move
                if eval == "mistake":
                    
                    # Increase the mistake counter to see if all moves were mistakes
                    mistakeCount += 1
                    
                    # Reset the move
                    board.unmakeMove(move, cell)
                    
                    # Remove the move from the set of moves, since it will be a taboo move in the real game
                    newMoveset = all_moves.copy()
                    newMoveset.remove((move, cell))
                    
                    # Make a new key for the board, since the taboo moves have changed
                    board.key ^= board.keyGenerator[move.i][move.j][move.value-1]
                    
                    # Run minimax from the others turn on the current state,
                    # with a little worse scoring to not encourage this when having multiple options
                    eval = self.minimax(board, tabooMoves, depth-1, False, score-(1/2),
                                        alpha, beta, newMoveset, moveListDict)
                    
                    # Reset the key of the board
                    board.key ^= board.keyGenerator[move.i][move.j][move.value-1]
                    
                    # If this move is also a mistake, then the current move is a mistake
                    if eval == "mistake":
                        return "mistake"
                    
                    # Make the move to make sure later the reset goes well
                    board.makeMove(move, cell)
                
                # If no mistake was made update the best score if needed and update alpha for the alpha beta pruning
                bestscore = max(bestscore, eval)
                alpha = max(alpha, eval)
                
                # If pruning is possible break out the loop and prune
                if beta <= alpha:
                    board.unmakeMove(move, cell)
                    break
                board.unmakeMove(move, cell)
        else:
            
            # Initializing the best score given minimizing player
            bestscore = math.inf
            
            # Running through all moves
            for move, cell in all_moves:
                
                # Make this move and update the board, it returns the amount of regions filled when making this move
                regionsFilled = board.makeMove(move, cell)
                
                # Check if we already computed the set of moves for this state,
                # if so retrive it from the dictionary, otherwise compute it
                subsetMoves = 0
                if board.key in moveListDict:
                    subsetMoves = moveListDict.get(board.key)
                else:
                    subsetMoves = self.getSubset(all_moves, move, cell)
                    moveListDict[board.key] = subsetMoves
                
                # Run the minimax as maximizing player
                eval = self.minimax(board, tabooMoves, depth-1, True, score-self.pointScore[regionsFilled],
                                    alpha, beta, subsetMoves, moveListDict)
                
                # If this move was a mistake we check if waiting in this possition is better, by playing this mistaken move
                if eval == "mistake":
                    
                    # Increase the mistake counter to see if all moves were mistakes
                    mistakeCount += 1
                    
                    # Reset the move
                    board.unmakeMove(move, cell)
                    
                    # Remove the move from the set of moves, since it will be a taboo move in the real game
                    newMoveset = all_moves.copy()
                    newMoveset.remove((move, cell))
                    
                    # Make a new key for the board, since the taboo moves have changed
                    board.key ^= board.keyGenerator[move.i][move.j][move.value-1]
                    
                    # Run minimax from the others turn on the current state,
                    # with a little worse scoring to not encourage this when having multiple options
                    eval = self.minimax(board, tabooMoves, depth-1, True, score+(1/2),
                                        alpha, beta, newMoveset, moveListDict)
                    
                    # Reset the key of the board
                    board.key ^= board.keyGenerator[move.i][move.j][move.value-1]
                    
                    # If this move is also a mistake, then the current move is a mistake
                    if eval == "mistake":
                        return "mistake"
                    
                    # Make the move to make sure later the reset goes well
                    board.makeMove(move, cell)
                    
                # If no mistake was made update the best score if needed and update alpha for the alpha beta pruning
                bestscore = min(bestscore, eval)
                beta = min(beta, eval)
                
                # If pruning is possible break out the loop and prune
                if beta <= alpha:
                    board.unmakeMove(move, cell)
                    break
                board.unmakeMove(move, cell)
        
        # If all moves were mistakes than a previous played move was a mistake, so return a mistake token
        if mistakeCount >= len(all_moves):
            return "mistake"
        
        # If no mistake was detected return the best score
        return bestscore
    
    
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

