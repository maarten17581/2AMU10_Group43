#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import random
import time
import math
import copy
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai
from team43_A1_better_search.regionsudokuboard import RegionSudokuBoard
from typing import List


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration.
    """
    
    pointScore = [0,1,3,7]

    def __init__(self):
        super().__init__()

    # N.B. This is a very naive implementation.
    def compute_best_move(self, game_state: GameState) -> None:
        N = game_state.board.N
        n = game_state.board.n
        m = game_state.board.m
        board = RegionSudokuBoard(game_state)
        all_moves = board.get_moves(game_state.taboo_moves)
        random.shuffle(all_moves)
        randomMove = random.choice(all_moves)[0]
        self.propose_move(randomMove)
        depth = 0
        bestscore = -math.inf
        bestmove = randomMove
        while depth <= board.emptyCount:
            bestscoreThisDepth = -math.inf
            bestmoveThisDepth = randomMove
            alpha = -math.inf
            beta = math.inf
            for move, cell in all_moves:
                regionsFilled = board.makeMove(move, cell)
                moveList = copy.deepcopy(all_moves)
                eval = self.minimax(board, game_state.taboo_moves, depth, False, 
                                        game_state.scores[0]-game_state.scores[1]+self.pointScore[regionsFilled],
                                        alpha, beta)
                if eval == "mistake":
                    board.unmakeMove(move, cell)
                    continue
                if eval > bestscoreThisDepth:
                    bestmoveThisDepth = move
                    bestscoreThisDepth = eval
                alpha = max(alpha, eval)
                if beta <= alpha:
                    board.unmakeMove(move, cell)
                    break
                board.unmakeMove(move, cell)
            self.propose_move(bestmoveThisDepth)
            #print(depth)
            depth += 1
            
    def minimax(self, board: RegionSudokuBoard, tabooMoves: List[TabooMove], depth: int, maximizingPlayer: bool, score: int, alpha, beta):
        N = board.N
        n = board.n
        m = board.m
        if depth == 0 or board.emptyCount == 0:
            return score
        bestscore = -math.inf if maximizingPlayer else math.inf
        all_moves = board.get_moves(tabooMoves)
        random.shuffle(all_moves)
        if len(all_moves) == 0:
            return "mistake"
        mistakeCount = 0
        for move, cell in all_moves:
            regionsFilled = board.makeMove(move, cell)
            if maximizingPlayer:
                eval = self.minimax(board, tabooMoves, depth-1, False, score+self.pointScore[regionsFilled], alpha, beta)
                if eval == "mistake":
                    mistakeCount += 1
                else:
                    bestscore = max(bestscore, eval)
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        board.unmakeMove(move, cell)
                        break
            else:
                eval = self.minimax(board, tabooMoves, depth-1, True, score-self.pointScore[regionsFilled], alpha, beta)
                if eval == "mistake":
                    mistakeCount += 1
                else:
                    bestscore = min(bestscore, eval)
                    beta = min(beta, eval)
                    if beta <= alpha:
                        board.unmakeMove(move, cell)
                        break
            board.unmakeMove(move, cell)
        if mistakeCount >= len(all_moves):
            return "mistake"
        return bestscore

