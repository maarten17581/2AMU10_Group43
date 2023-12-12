#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import random
import time
import math
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai
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
        board = game_state.board
        N = board.N
        n = board.n
        m = board.m
        first = True
        all_moves = []
        for i in range(N):
            for j in range(N):
                for value in range(1, N+1):
                    if SudokuAI.possible(i, j, value, board, game_state.taboo_moves, n, m, N):
                        if first:
                            first = False
                            self.propose_move(Move(i, j, value))
                        all_moves.append(Move(i, j, value))
        randomMove = random.choice(all_moves)
        self.propose_move(randomMove)
        depth = 0
        bestscore = -math.inf
        bestmove = randomMove
        while depth <= N*N:
            for move in all_moves:
                board.put(move.i, move.j, move.value)
                regions = 3
                for (k,l) in SudokuAI.row(move.i, move.j, n, m, N):
                    if SudokuBoard.get(board, k, l) == SudokuBoard.empty:
                        regions -= 1
                        break
                for (k,l) in SudokuAI.col(move.i, move.j, n, m, N):
                    if SudokuBoard.get(board, k, l) == SudokuBoard.empty:
                        regions -= 1
                        break
                for (k,l) in SudokuAI.box(move.i, move.j, n, m, N):
                    if SudokuBoard.get(board, k, l) == SudokuBoard.empty:
                        regions -= 1
                        break
                eval = SudokuAI.minimax(board, game_state.taboo_moves, depth, False, 
                                        game_state.scores[0]-game_state.scores[1]+SudokuAI.pointScore[regions])
                if eval > bestscore:
                    bestmove = move
                    bestscore = eval
                    self.propose_move(bestmove)
                board.put(move.i, move.j, board.empty)
            #print(depth)
            depth += 1
            
    def minimax(board: SudokuBoard, tabooMoves: List[TabooMove], depth: int, maximizingPlayer: bool, score: int):
        N = board.N
        n = board.n
        m = board.m
        all_moves = SudokuAI.possibleMoves(board, tabooMoves)
        if depth == 0:
            return score
        bestscore = -math.inf if maximizingPlayer else math.inf
        for move in all_moves:
            board.put(move.i, move.j, move.value)
            regions = 3
            for (k,l) in SudokuAI.row(move.i, move.j, n, m, N):
                if SudokuBoard.get(board, k, l) == SudokuBoard.empty:
                    regions -= 1
                    break
            for (k,l) in SudokuAI.col(move.i, move.j, n, m, N):
                if SudokuBoard.get(board, k, l) == SudokuBoard.empty:
                    regions -= 1
                    break
            for (k,l) in SudokuAI.box(move.i, move.j, n, m, N):
                if SudokuBoard.get(board, k, l) == SudokuBoard.empty:
                    regions -= 1
                    break
            if maximizingPlayer:
                eval = SudokuAI.minimax(board, tabooMoves, depth-1, False, score+SudokuAI.pointScore[regions])
                bestscore = max(bestscore, eval)
            else:
                eval = SudokuAI.minimax(board, tabooMoves, depth-1, True, score-SudokuAI.pointScore[regions])
                bestscore = min(bestscore, eval)
            board.put(move.i, move.j, board.empty)
        return bestscore
                
            
    def row(i, j, n, m, N):
        return [(i,k) for k in range(N)]
            
    def col(i, j, n, m, N):
        return [(k,j) for k in range(N)]
        
    def box(i, j, n, m, N):
        return [(k,l) for k in range((i//m)*m, (i//m)*m+m) for l in range((j//n)*n, (j//n)*n+n)]
    
    def possible(i, j, value, board, tabooMoves, n, m, N):
        if board.get(i, j) != SudokuBoard.empty \
                or TabooMove(i, j, value) in tabooMoves:
            return False
        #print("possible "+str(i)+" "+str(j)+" "+str(value))
        for (k,l) in SudokuAI.row(i,j, n, m, N):
            #print("row "+str(k)+" "+str(l))
            if board.get(k,l) == value and (k != i or l != j):
                return False
        for (k,l) in SudokuAI.col(i,j, n, m, N):
            #print("col "+str(k)+" "+str(l))
            if board.get(k,l) == value and (k != i or l != j):
                return False
        for (k,l) in SudokuAI.box(i,j, n, m, N):
            #print("box "+str(k)+" "+str(l))
            if board.get(k,l) == value and (k != i or l != j):
                return False
        return True
    
    def possibleMoves(board: SudokuBoard, tabooMoves: List[TabooMove]) -> List[Move]:
        N = board.N
        n = board.n
        m = board.m
        moves = []
        for i in range(N):
            for j in range(N):
                for value in range(1, N+1):
                    if SudokuAI.possible(i, j, value, board, tabooMoves, n, m, N):
                        moves.append(Move(i, j, value))
        return moves

