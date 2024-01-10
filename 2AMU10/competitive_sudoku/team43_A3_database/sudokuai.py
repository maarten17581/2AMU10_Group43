#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

from datetime import datetime
import random
import time
import math
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai
from team43_A3_database.regionsudokuboard import RegionSudokuBoard
from typing import List
import copy


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
        The implementation of the AI Agent for Assignment 3
        @param game_state: an instance of the game provided by the simulator
        """
        
        N = game_state.board.N
        n = game_state.board.n
        m = game_state.board.m
        
        # Create our custom board data structure from the current board
        board = RegionSudokuBoard(game_state)
        
        startBoardHash = board.getCanonicalHash()
        startBoardKey = board.key
        startMoveHash = board.getMoveHash(startBoardKey, game_state.taboo_moves)

        # Create all moves for the first time using the board
        all_moves, _ = board.get_moves(game_state.taboo_moves)

        # Pick a random move for now
        randomMove = random.choice(all_moves)[0]
        self.propose_move(randomMove)
        
        # The starting difference between the scores of the current state seeing our player as maximizing player
        dif = game_state.scores[0]-game_state.scores[1] if game_state.current_player() == 1 else game_state.scores[1]-game_state.scores[0]
        
        nodeTable = {}
        
        bestScore = -1
        bestMove = None
        bestIsMistake = None
        addCount = 0
        # Our iteritive deepening loop, if the depth is more than the number of empty squares it does not improve anymore
        while True:
            scoredif = game_state.scores[game_state.current_player()-1]-game_state.scores[2-game_state.current_player()]

            # Run monte carlo tree search
            winrate, visits, move, mistake = self.mcts(nodeTable, board, game_state.taboo_moves, scoredif, True)
            #print(f"{winrate}, {visits}, {move}, {mistake}")
            # Propose the best move for this depth and go to the next
            if move != None:
                boardHash = startBoardHash
                moveHash = 0
                if mistake:
                    new_taboo_moves = copy.copy(game_state.taboo_moves)
                    new_taboo_moves.append(TabooMove(move[0].i, move[0].j, move[0].value))
                    moveHash = board.getMoveHash(board.key, new_taboo_moves)
                else:
                    boardHash = board.getCanonicalHashAfterMove(move[0])
                    board.makeMove(move[0], move[1], False)
                    moveHash = board.getMoveHash(board.key, game_state.taboo_moves)
                    board.unmakeMove(move[0], move[1])
                
                #print(f"{nodeTable[(boardHash, moveHash)][0]/nodeTable[(boardHash, moveHash)][1] > bestScore} {(nodeTable[(boardHash, moveHash)][0]/nodeTable[(boardHash, moveHash)][1] == bestScore and (board.emptyCount%2 == 0) == mistake)} {(nodeTable[(boardHash, moveHash)][0]/nodeTable[(boardHash, moveHash)][1] == bestScore and (board.emptyCount%2 == 0) != mistake and bestIsMistake != None and (board.emptyCount%2 == 0) != bestIsMistake and board.regionsFilledAfterMove(move[0]) > board.regionsFilledAfterMove(bestMove[0]))}")
                if bestMove != move and (nodeTable[(boardHash, moveHash)][0]/nodeTable[(boardHash, moveHash)][1] > bestScore or (nodeTable[(boardHash, moveHash)][0]/nodeTable[(boardHash, moveHash)][1] == bestScore and (board.emptyCount%2 == 0) == mistake) or (nodeTable[(boardHash, moveHash)][0]/nodeTable[(boardHash, moveHash)][1] == bestScore and (board.emptyCount%2 == 0) != mistake and bestIsMistake != None and (board.emptyCount%2 == 0) != bestIsMistake and move[1].rowRegion.filled+move[1].colRegion.filled+move[1].boxRegion.filled < bestMove[1].rowRegion.filled+bestMove[1].colRegion.filled+bestMove[1].boxRegion.filled)):
                    #print(board.emptyCount)
                    bestScore = nodeTable[(boardHash, moveHash)][0]/nodeTable[(boardHash, moveHash)][1]
                    bestMove = move
                    bestIsMistake = mistake
                    #print(f"{bestScore} {bestMove[0].i} {bestMove[0].j} {bestMove[0].value} {mistake}")
                    self.propose_move(bestMove[0])
                elif bestMove == move:
                    #print("###")
                    bestScore = -1
                    bestMove = None
                    boardHash = board.getCanonicalHash()
                    nowMoveHash = board.getMoveHash(board.key, game_state.taboo_moves)
                    for newMove, childHash in nodeTable[(boardHash, nowMoveHash)][2]:
                        #print(f"{newMove[0].i} {newMove[0].j} {newMove[0].value} {nodeTable[childHash][0]/nodeTable[childHash][1]}")
                        childMistake = (childHash[0] == boardHash)
                        #if childMistake:
                            #print(f"{newMove[0]} <---------")
                        if nodeTable[childHash][0]/nodeTable[childHash][1] > bestScore or (nodeTable[childHash][0]/nodeTable[childHash][1] == bestScore and (board.emptyCount%2 == 0) == childMistake) or (nodeTable[childHash][0]/nodeTable[childHash][1] == bestScore and (board.emptyCount%2 == 0) != childMistake and bestIsMistake != None and (board.emptyCount%2 == 0) != bestIsMistake and newMove[1].rowRegion.filled+newMove[1].colRegion.filled+newMove[1].boxRegion.filled < bestMove[1].rowRegion.filled+bestMove[1].colRegion.filled+bestMove[1].boxRegion.filled):
                            bestScore = nodeTable[childHash][0]/nodeTable[childHash][1]
                            bestMove = newMove
                            bestIsMistake = childMistake
                    #print(f"{bestScore} {bestMove[0].i} {bestMove[0].j} {bestMove[0].value} {mistake}")
                    #print("###")
                self.propose_move(bestMove[0])
            addCount+=1
            #print(f"{addCount} {bestScore} {bestMove[0] if bestMove != None else 'None'} {bestIsMistake} {move[0] if move != None else 'None'} {mistake}")
            #if (startBoardHash, startMoveHash) in nodeTable:
            #    print(f"{board.emptyCount} {len(nodeTable[(startBoardHash, startMoveHash)][2])}")
    
    def mcts(self, nodeTable, board: RegionSudokuBoard, taboo_moves, scoredif, maximizing_player):
        boardHash = board.getCanonicalHash()
        moveHash = board.getMoveHash(board.key, taboo_moves)
        if (boardHash, moveHash) in nodeTable:
            print("do mcts step")
            possible_moves, mistake_moves = board.get_moves(taboo_moves)
            if len(possible_moves) == 0:
                if board.emptyCount != 0:
                    return "mistake", None, None, None, None
                else:
                    if scoredif > 0:
                        added_score = -1 if maximizing_player else 1
                        nodeTable[(boardHash, moveHash)] = (nodeTable[(boardHash, moveHash)][0] + added_score, nodeTable[(boardHash, moveHash)][1]+1, nodeTable[(boardHash, moveHash)][2], nodeTable[(boardHash, moveHash)][3])
                        return 1, 1, None, False, scoredif
                    else:
                        nodeTable[(boardHash, moveHash)] = (nodeTable[(boardHash, moveHash)][0], nodeTable[(boardHash, moveHash)][1]+1, nodeTable[(boardHash, moveHash)][2], nodeTable[(boardHash, moveHash)][3])
                        return 0, 1, None, False, scoredif
            random.shuffle(possible_moves)
            max_possible = max(possible_moves, key=lambda move: self.uct(board, move, nodeTable, False, taboo_moves))
            uct_possible = self.uct(board, max_possible, nodeTable, False, taboo_moves)
            uct_mistake = -math.inf
            if len(mistake_moves) > 0:
                random.shuffle(mistake_moves)
                max_mistake = max(mistake_moves, key=lambda move: self.uct(board, move, nodeTable, True, taboo_moves))
                uct_mistake = self.uct(board, max_mistake, nodeTable, True, taboo_moves)
            while uct_possible >= uct_mistake:
                score = board.makeMove(max_possible[0], max_possible[1], True)
                if score != "mistake":
                    score_added = self.pointScore[score] if maximizing_player else -self.pointScore[score]
                    winrate, visits, move, mistake, endedScoreDif = self.mcts(nodeTable, board, taboo_moves, scoredif+score_added, not maximizing_player)
                    board.unmakeMove(max_possible[0], max_possible[1])
                    if winrate == "mistake":
                        possible_moves.remove(max_possible)
                        mistake_moves.append(max_possible)
                        board.makeMistake(max_possible)
                        if len(possible_moves) == 0:
                            return "mistake", None, None, None, None
                        max_possible = max(possible_moves, key=lambda move: self.uct(board, move, nodeTable, False, taboo_moves))
                        max_mistake = max(mistake_moves, key=lambda move: self.uct(board, move, nodeTable, True, taboo_moves))
                        uct_possible = self.uct(board, max_possible, nodeTable, False, taboo_moves)
                        uct_mistake = self.uct(board, max_mistake, nodeTable, True, taboo_moves)
                    else:
                        childList = nodeTable[(boardHash, moveHash)][2]
                        if move == None:
                            childHash = board.getCanonicalHashAfterMove(max_possible[0])
                            board.makeMove(max_possible[0], max_possible[1], False)
                            childMoveHash = board.getMoveHash(board.key, taboo_moves)
                            childList.append((max_possible, (childHash, childMoveHash)))
                            board.unmakeMove(max_possible[0], max_possible[1])
                        scoredifList = nodeTable[(boardHash, moveHash)][3]
                        scoredifList.append(endedScoreDif-scoredif)
                        if maximizing_player:
                            nodeTable[(boardHash, moveHash)] = (nodeTable[(boardHash, moveHash)][0]-winrate, nodeTable[(boardHash, moveHash)][1]+visits, childList, scoredifList)
                        else:
                            nodeTable[(boardHash, moveHash)] = (nodeTable[(boardHash, moveHash)][0]+winrate, nodeTable[(boardHash, moveHash)][1]+visits, childList, scoredifList)
                        return winrate, visits, max_possible, False, endedScoreDif
                else:
                    board.unmakeMove(max_possible[0], max_possible[1])
                    possible_moves.remove(max_possible)
                    mistake_moves.append(max_possible)
                    board.makeMistake(max_possible)
                    if len(possible_moves) == 0:
                        return "mistake", None, None, None, None
                    max_possible = max(possible_moves, key=lambda move: self.uct(board, move, nodeTable, False, taboo_moves))
                    max_mistake = max(mistake_moves, key=lambda move: self.uct(board, move, nodeTable, True, taboo_moves))
                    uct_possible = self.uct(board, max_possible, nodeTable, False, taboo_moves)
                    uct_mistake = self.uct(board, max_mistake, nodeTable, True, taboo_moves)
            if uct_possible < uct_mistake:
                board.key = board.newHashOfMove(max_mistake[0], True)
                new_taboo_list = copy.copy(taboo_moves)
                new_taboo_list.append(TabooMove(max_mistake[0].i, max_mistake[0].j, max_mistake[0].value))
                winrate, visits, move, mistake, endedScoreDif = self.mcts(nodeTable, board, new_taboo_list, scoredif, not maximizing_player)
                board.key = board.newHashOfMove(max_mistake[0], True)
                if winrate == "mistake":
                    return "mistake", None, None, None, None
                childList = nodeTable[(boardHash, moveHash)][2]
                if move == None:
                    childList.append((max_mistake, (boardHash, board.getMoveHash(board.key, new_taboo_list))))
                scoredifList = nodeTable[(boardHash, moveHash)][3]
                scoredifList.append(endedScoreDif-scoredif)
                if maximizing_player:
                    nodeTable[(boardHash, moveHash)] = (nodeTable[(boardHash, moveHash)][0]-winrate, nodeTable[(boardHash, moveHash)][1]+visits, childList, scoredifList)
                else:
                    nodeTable[(boardHash, moveHash)] = (nodeTable[(boardHash, moveHash)][0]+winrate, nodeTable[(boardHash, moveHash)][1]+visits, childList, scoredifList)
                return winrate, visits, max_mistake, True, endedScoreDif
        else:
            print("start simulation")
            winrate, endedScoreDif = self.simulation(board, taboo_moves, scoredif, maximizing_player)
            if winrate == "mistake":
                return "mistake", None, None, None, None
            if maximizing_player:
                nodeTable[(boardHash, moveHash)] = (-winrate, 1, [], [endedScoreDif-scoredif])
            else:
                nodeTable[(boardHash, moveHash)] = (winrate, 1, [], [endedScoreDif-scoredif])
            print("start back prop")
            return winrate, 1, None, None, endedScoreDif

    def uct(self, board: RegionSudokuBoard, move, nodeTable, mistake, taboo_moves):
        nowBoardHash = board.getCanonicalHash()
        nextBoardHash = nowBoardHash
        nowMoveHash = board.getMoveHash(board.key, taboo_moves)
        if not mistake:
            nextBoardHash = board.getCanonicalHashAfterMove(move[0])
        nextMoveHash = 0
        if mistake:
            new_taboo_moves = copy.copy(taboo_moves)
            new_taboo_moves.append(TabooMove(move[0].i, move[0].j, move[0].value))
            nextMoveHash = board.getMoveHash(board.key, new_taboo_moves)
        else:
            board.makeMove(move[0], move[1], False)
            nextMoveHash = board.getMoveHash(board.key, taboo_moves)
            board.unmakeMove(move[0], move[1])
        
        if (nextBoardHash, nextMoveHash) not in nodeTable or nodeTable[(nextBoardHash, nextMoveHash)][1] == 0:
            return math.inf
        return (nodeTable[(nextBoardHash, nextMoveHash)][0]/nodeTable[(nextBoardHash, nextMoveHash)][1]+3*math.sqrt(math.log(nodeTable[(nowBoardHash, nowMoveHash)][1])/nodeTable[(nextBoardHash, nextMoveHash)][1]))
    
    def simulation(self, board: RegionSudokuBoard, taboo_moves, scoredif, maximizing_player):
        #print(f"sim depth {board.emptyCount} taboo length {len(taboo_moves)}")
        if board.emptyCount == 0:
            if scoredif > 0:
                return 1, scoredif
            else:
                return 0, scoredif
        #if scoredif > (board.emptyCount/2):
        #    return 1
        #elif scoredif < -(board.emptyCount/2):
        #    return 0
        winrate = "mistake"
        possible_moves, mistake_moves = board.get_moves(taboo_moves)
        while True:
            #print(board.emptyCount)
            if len(possible_moves) == 0:
                return "mistake", None
            madeMove = None
            if len(mistake_moves)%2 == 1:
                if random.random() < 0.9:
                    madeMove = max(possible_moves+mistake_moves, key=lambda move: (board.regionsFilledAfterMove(move[0]), -move[1].valueCount, move[1].rowRegion.filled+move[1].colRegion.filled+move[1].boxRegion.filled))
                else:
                    madeMove = random.choice(possible_moves+mistake_moves)
            else:
                if random.random() < 0.9:
                    madeMove = max(possible_moves, key=lambda move: (board.regionsFilledAfterMove(move[0]), -move[1].valueCount, move[1].rowRegion.filled+move[1].colRegion.filled+move[1].boxRegion.filled))
                else:
                    madeMove = random.choice(possible_moves)
            score = 0
            if madeMove in possible_moves:
                score = board.makeMove(madeMove[0], madeMove[1], True)
            else:
                board.key = board.newHashOfMove(madeMove[0], True)
            if score == "mistake":
                board.unmakeMove(madeMove[0], madeMove[1])
                possible_moves.remove(madeMove)
                mistake_moves.append(madeMove)
                board.makeMistake(madeMove)
                board.key = board.newHashOfMove(madeMove[0], True)
                score = 0
            new_taboo_list = copy.copy(taboo_moves)
            if madeMove in mistake_moves:
                new_taboo_list.append(TabooMove(madeMove[0].i, madeMove[0].j, madeMove[0].value))
            
            if maximizing_player:
                winrate, endedScoreDif = self.simulation(board, new_taboo_list, scoredif+self.pointScore[score], not maximizing_player)
            else:
                winrate, endedScoreDif = self.simulation(board, new_taboo_list, scoredif-self.pointScore[score], not maximizing_player)
            if madeMove in mistake_moves:
                board.key = board.newHashOfMove(madeMove[0], True)
            else:
                board.unmakeMove(madeMove[0], madeMove[1])
            if winrate == "mistake":
                if madeMove in mistake_moves:
                    return winrate, None
                else:
                    possible_moves.remove(madeMove)
                    mistake_moves.append(madeMove)
                    board.makeMistake(madeMove)
            else:
                return winrate, endedScoreDif