#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import random
import time
import math
from datetime import datetime
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai
from team43_A3_NN.regionsudokuboard import RegionSudokuBoard
from team43_A3_NN.neuralnetwork import NeuralNetwork
from typing import List
import numpy as np

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
        
        # Get the neural network
        neural_net = NeuralNetwork(None, "trained_weights.txt")
        
        # Get the output of the neural network for the given input
        nn_output = neural_net.call(self.input_vector(board, game_state.taboo_moves))
        
        # Get the best move from the output and determine which move it is
        best_index = nn_output.argmax()
        best_i = best_index//(N**2)
        best_j = (best_index%(N**2))//N
        best_value = best_index%N+1
        
        # Propose that move
        self.propose_move(Move(best_i, best_j, best_value))
            
    def input_vector(self, board: RegionSudokuBoard, taboo_moves):
        input = [0]*(2*(board.N**3))
        for cell in board.cells:
            if cell.value != 0:
                input[cell.i*(board.N**2)+cell.j*board.N+cell.value-1] = 1
        possibleMoves, mistakeMoves = board.get_moves(taboo_moves)
        moves = [move for move in possibleMoves + mistakeMoves if TabooMove(move[0].i, move[0].j, move[0].value) not in taboo_moves]
        for move, cell in moves:
            input[board.N**3+move.i*(board.N**2)+move.j*board.N+move.value-1] = 1
        return input
