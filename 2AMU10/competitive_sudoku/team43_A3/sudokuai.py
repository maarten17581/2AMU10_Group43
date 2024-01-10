#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

from datetime import datetime
from io import TextIOWrapper
import random
import time
import math
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai
from team43_A3.regionsudokuboard import RegionSudokuBoard
from typing import List
import copy
import json

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
        
        # The table that is computed beforehand
        preNodeTable = {}
        
        # Load in the table and the corresponding hashing tables, that are used for the keys of the table
        if m == 3 and n == 3:
            keyGenerator = json.loads(self.keyGenerator3x3)
            moveKeyGenerator = json.loads(self.moveKeyGenerator3x3)
            board.keyGenerator = keyGenerator
            board.moveKeyGenerator = moveKeyGenerator
            
            # Recompute the regular board hash with the new hash tables
            board.key = 0
            for i in range(N):
                for j in range(N):
                    
                    # Get the value that this cell has in the sudoku
                    value = game_state.board.get(i, j)
                    
                    # Add the key value to the key using a bitwise XOR to update the key
                    board.key ^= board.keyGenerator[i][j][value]
            jsonPreNodeTable = json.loads(self.nodeTable3x3)
            preNodeTable = self.jsonTableToNormal(jsonPreNodeTable, board)
        
        # Get the hashes of the canonical board for the current position
        startBoardHash = board.getCanonicalHash()
        startBoardKey = board.key
        startMoveHash = board.getMoveHash(startBoardKey, game_state.taboo_moves)
        
        # Create all moves for the first time using the board
        all_moves, _ = board.get_moves(game_state.taboo_moves)
        
        # Pick a random move for now
        randomMove = random.choice(all_moves)
        self.propose_move(randomMove[0])
        
        # The starting difference between the scores of the current state seeing our player as maximizing player
        dif = game_state.scores[0]-game_state.scores[1] if game_state.current_player() == 1 else game_state.scores[1]-game_state.scores[0]
        
        # Initiallize our regular search tree
        nodeTable = {}
        
        # Initialize the best scores and moves that are saved during iterations
        bestScore = -1
        bestMove = None
        bestIsMistake = None
        
        # Keep iterating untill the time is over
        while True:
            # Calculate the difference between the scores
            scoredif = game_state.scores[game_state.current_player()-1]-game_state.scores[2-game_state.current_player()]

            # Run monte carlo tree search
            winrate, visits, move, mistake = self.mcts(nodeTable, board, game_state.taboo_moves, scoredif, True, preNodeTable)
            
            # If a move was computed, compute if this move is better then what we had
            if move != None:
                boardHash = startBoardHash
                moveHash = 0
                if mistake:
                    new_taboo_moves = copy.copy(game_state.taboo_moves)
                    new_taboo_moves.append(TabooMove(move[0].i, move[0].j, move[0].value))
                    moveHash = board.getMoveHash(board.key, new_taboo_moves)
                else:
                    boardHash = board.getCanonicalHashAfterMove(move[0])
                    board.makeMove(move[0], move[1])
                    moveHash = board.getMoveHash(board.key, game_state.taboo_moves)
                    board.unmakeMove(move[0], move[1])
                
                # If this move is better or whenever it is equal and the this move makes sure 
                # that we are the last playing player make this the best move
                if bestMove != move and \
                        (nodeTable[(boardHash, moveHash)][0]/nodeTable[(boardHash, moveHash)][1] > bestScore or \
                        (nodeTable[(boardHash, moveHash)][0]/nodeTable[(boardHash, moveHash)][1] == bestScore and \
                        (board.emptyCount%2 == 0) == mistake) or \
                        (nodeTable[(boardHash, moveHash)][0]/nodeTable[(boardHash, moveHash)][1] == bestScore \
                        and (board.emptyCount%2 == 0) != mistake and bestIsMistake != None \
                        and (board.emptyCount%2 == 0) != bestIsMistake and \
                        move[1].rowRegion.filled+move[1].colRegion.filled+move[1].boxRegion.filled \
                        < bestMove[1].rowRegion.filled+bestMove[1].colRegion.filled+bestMove[1].boxRegion.filled)):
                    
                    bestScore = nodeTable[(boardHash, moveHash)][0]/nodeTable[(boardHash, moveHash)][1]
                    bestMove = move
                    bestIsMistake = mistake
                    
                    self.propose_move(bestMove[0])
                # If this move was already the best move and we went in value, recompute the best move for all moves
                elif bestMove == move:
                    
                    # Initialize the best moves again
                    bestScore = -1
                    bestMove = None
                    boardHash = board.getCanonicalHash()
                    nowMoveHash = board.getMoveHash(board.key, game_state.taboo_moves)
                    
                    # Check for each child if it is better
                    for newMove, childHash in nodeTable[(boardHash, nowMoveHash)][2]:
                        
                        childMistake = (childHash[0] == boardHash)
                        
                        # If this move is better or whenever it is equal and the this move makes sure 
                        # that we are the last playing player make this the best move
                        if nodeTable[childHash][0]/nodeTable[childHash][1] > bestScore or \
                                (nodeTable[childHash][0]/nodeTable[childHash][1] == bestScore and \
                                (board.emptyCount%2 == 0) == childMistake) or \
                                (nodeTable[childHash][0]/nodeTable[childHash][1] == bestScore and \
                                (board.emptyCount%2 == 0) != childMistake and bestIsMistake != None and \
                                (board.emptyCount%2 == 0) != bestIsMistake and \
                                newMove[1].rowRegion.filled+newMove[1].colRegion.filled+newMove[1].boxRegion.filled \
                                < bestMove[1].rowRegion.filled+bestMove[1].colRegion.filled+bestMove[1].boxRegion.filled):
                            bestScore = nodeTable[childHash][0]/nodeTable[childHash][1]
                            bestMove = newMove
                            bestIsMistake = childMistake
                    
                self.propose_move(bestMove[0])
    
    def mcts(self, nodeTable, board: RegionSudokuBoard, taboo_moves, scoredif, maximizing_player, preNodeTable):
        """
        Runs monte carlo tree search
        """
        boardHash = board.getCanonicalHash()
        moveHash = board.getMoveHash(board.key, taboo_moves)
        
        # If it was already in the node table, then we are in the leaf selection fase
        if (boardHash, moveHash) in nodeTable:
            possible_moves, mistake_moves = board.get_moves(taboo_moves)
            
            # If there are no moves possible, we are either in a mistake,
            # if the board has empty cells left, or at the end. We return the appropriate response
            if len(possible_moves) == 0:
                if board.emptyCount != 0:
                    return "mistake", None, None, None
                else:
                    if scoredif > 0:
                        added_score = -1 if maximizing_player else 1
                        nodeTable[(boardHash, moveHash)] = (nodeTable[(boardHash, moveHash)][0] + added_score, nodeTable[(boardHash, moveHash)][1]+1, nodeTable[(boardHash, moveHash)][2])
                        return 1, 1, None, False
                    else:
                        nodeTable[(boardHash, moveHash)] = (nodeTable[(boardHash, moveHash)][0], nodeTable[(boardHash, moveHash)][1]+1, nodeTable[(boardHash, moveHash)][2])
                        return 0, 1, None, False
            random.shuffle(possible_moves)
            
            # Get the best possible move and best mistake move (if there are any) based on the UCT
            max_possible = max(possible_moves, key=lambda move: self.uct(board, move, nodeTable, False, taboo_moves))
            uct_possible = self.uct(board, max_possible, nodeTable, False, taboo_moves)
            uct_mistake = -math.inf
            if len(mistake_moves) > 0:
                random.shuffle(mistake_moves)
                max_mistake = max(mistake_moves, key=lambda move: self.uct(board, move, nodeTable, True, taboo_moves))
                uct_mistake = self.uct(board, max_mistake, nodeTable, True, taboo_moves)
                
            # If the possible move was better then the mistake move we make that move
            while uct_possible >= uct_mistake:
                score = board.makeMove(max_possible[0], max_possible[1])
                if score != "mistake":
                    
                    # We run mcts after making this move
                    score_added = self.pointScore[score] if maximizing_player else -self.pointScore[score]
                    winrate, visits, move, mistake = self.mcts(nodeTable, board, taboo_moves, scoredif+score_added, not maximizing_player, preNodeTable)
                    board.unmakeMove(max_possible[0], max_possible[1])
                    
                    # If this lead to a mistake, then the made move was a mistake and we add it to the mistake moves
                    if winrate == "mistake":
                        possible_moves.remove(max_possible)
                        mistake_moves.append(max_possible)
                        board.makeMistake(max_possible)
                        if len(possible_moves) == 0:
                            return "mistake", None, None, None 
                        max_possible = max(possible_moves, key=lambda move: self.uct(board, move, nodeTable, False, taboo_moves))
                        max_mistake = max(mistake_moves, key=lambda move: self.uct(board, move, nodeTable, True, taboo_moves))
                        uct_possible = self.uct(board, max_possible, nodeTable, False, taboo_moves)
                        uct_mistake = self.uct(board, max_mistake, nodeTable, True, taboo_moves)
                        
                    else: # If it wasnt a mistake, then we are now in the back propagation phase, 
                          # because we are returning from the mcts. Therefore update this node and return the winrate and visits
                        
                        childList = nodeTable[(boardHash, moveHash)][2]
                        if move == None:
                            childHash = board.getCanonicalHashAfterMove(max_possible[0])
                            board.makeMove(max_possible[0], max_possible[1])
                            childMoveHash = board.getMoveHash(board.key, taboo_moves)
                            childList.append((max_possible, (childHash, childMoveHash)))
                            board.unmakeMove(max_possible[0], max_possible[1])
                        if maximizing_player:
                            nodeTable[(boardHash, moveHash)] = (nodeTable[(boardHash, moveHash)][0]-winrate, nodeTable[(boardHash, moveHash)][1]+visits, childList)
                        else:
                            nodeTable[(boardHash, moveHash)] = (nodeTable[(boardHash, moveHash)][0]+winrate, nodeTable[(boardHash, moveHash)][1]+visits, childList)
                        return winrate, visits, max_possible, False
                    
                else: # Here the move we made was a mistake, so we add it to the mistake moves list 
                      # and recompute the best move to make
                    board.unmakeMove(max_possible[0], max_possible[1])
                    possible_moves.remove(max_possible)
                    mistake_moves.append(max_possible)
                    board.makeMistake(max_possible)
                    if len(possible_moves) == 0:
                        return "mistake", None, None, None
                    max_possible = max(possible_moves, key=lambda move: self.uct(board, move, nodeTable, False, taboo_moves))
                    max_mistake = max(mistake_moves, key=lambda move: self.uct(board, move, nodeTable, True, taboo_moves))
                    uct_possible = self.uct(board, max_possible, nodeTable, False, taboo_moves)
                    uct_mistake = self.uct(board, max_mistake, nodeTable, True, taboo_moves)
            
            # If the mistake move was better, then play that move
            if uct_possible < uct_mistake:
                
                # Run mcts form here after this move
                board.key = board.newHashOfMove(max_mistake[0], True)
                new_taboo_list = copy.copy(taboo_moves)
                new_taboo_list.append(TabooMove(max_mistake[0].i, max_mistake[0].j, max_mistake[0].value))
                winrate, visits, move, mistake = self.mcts(nodeTable, board, new_taboo_list, scoredif, not maximizing_player, preNodeTable)
                board.key = board.newHashOfMove(max_mistake[0], True)
                
                # If this move was a mistake, then the mistake has been made earlier, 
                # as this was already a mistake move. Thus return mistake
                if winrate == "mistake":
                    return "mistake", None, None, None
                
                # Update the table as we are now in the back propagation phase
                childList = nodeTable[(boardHash, moveHash)][2]
                if move == None:
                    childList.append((max_mistake, (boardHash, board.getMoveHash(board.key, new_taboo_list))))
                if maximizing_player:
                    nodeTable[(boardHash, moveHash)] = (nodeTable[(boardHash, moveHash)][0]-winrate, nodeTable[(boardHash, moveHash)][1]+visits, childList)
                else:
                    nodeTable[(boardHash, moveHash)] = (nodeTable[(boardHash, moveHash)][0]+winrate, nodeTable[(boardHash, moveHash)][1]+visits, childList)
                return winrate, visits, max_mistake, True
            
        else: # If this board was not in the tree, then we go to the simulation phase
            
            # If this position was already in the precomputed table, then return this instead of a simulation
            if (boardHash, moveHash) in preNodeTable:
                allRuns = preNodeTable[(boardHash, moveHash)][3]
                winrate = len([run for run in allRuns if run+scoredif>0])
                visits = len(allRuns)
                if maximizing_player:
                    nodeTable[(boardHash, moveHash)] = (-winrate, visits, [])
                else:
                    nodeTable[(boardHash, moveHash)] = (winrate, visits, [])
                return winrate, visits, None, None
            
            # Simulate a game from this position
            winrate = self.simulation(board, taboo_moves, scoredif, maximizing_player)
            
            # If this was a mistake, then the last played move was a mistake, so return mistake
            if winrate == "mistake":
                return "mistake", None, None, None
            
            # Update the table with the new node
            if maximizing_player:
                nodeTable[(boardHash, moveHash)] = (-winrate, 1, [])
            else:
                nodeTable[(boardHash, moveHash)] = (winrate, 1, [])
            
            # Return if we won and the visits
            return winrate, 1, None, None

    def uct(self, board: RegionSudokuBoard, move, nodeTable, mistake, taboo_moves):
        """
        Computes the UCT value for any move at any board position
        """
        # Get all the hashes
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
            board.makeMove(move[0], move[1])
            nextMoveHash = board.getMoveHash(board.key, taboo_moves)
            board.unmakeMove(move[0], move[1])
        
        # If it wasnt in the table yet or if the visit count is 0, return infinity
        if (nextBoardHash, nextMoveHash) not in nodeTable or nodeTable[(nextBoardHash, nextMoveHash)][1] == 0:
            return math.inf
        
        # Else just return the UCT based on the values in the table
        return (nodeTable[(nextBoardHash, nextMoveHash)][0]/nodeTable[(nextBoardHash, nextMoveHash)][1]+3*math.sqrt(math.log(nodeTable[(nowBoardHash, nowMoveHash)][1])/nodeTable[(nextBoardHash, nextMoveHash)][1]))
    
    def simulation(self, board: RegionSudokuBoard, taboo_moves, scoredif, maximizing_player):
        """
        Simulates a game
        """
        # If we are at the end, return if we have won
        if board.emptyCount == 0:
            if scoredif > 0:
                return 1
            else:
                return 0
        
        winrate = "mistake"
        possible_moves, mistake_moves = board.get_moves(taboo_moves)
        while True:
            # If there are no possible moves we made a mistake, return that
            if len(possible_moves) == 0:
                return "mistake"
            
            # Get a move, based on a greedy tactick or 10% of the time a random move
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
            
            # Make that move
            score = 0
            if madeMove in possible_moves:
                score = board.makeMove(madeMove[0], madeMove[1])
            else:
                board.key = board.newHashOfMove(madeMove[0], True)
            
            # If it leads to a mistake, add it to the list of mistake moves
            if score == "mistake":
                board.unmakeMove(madeMove[0], madeMove[1])
                possible_moves.remove(madeMove)
                mistake_moves.append(madeMove)
                board.makeMistake(madeMove)
                board.key = board.newHashOfMove(madeMove[0], True)
                score = 0
            
            # Run the simulation after making this move
            new_taboo_list = copy.copy(taboo_moves)
            if madeMove in mistake_moves:
                new_taboo_list.append(TabooMove(madeMove[0].i, madeMove[0].j, madeMove[0].value))
            
            if maximizing_player:
                winrate = self.simulation(board, new_taboo_list, scoredif+self.pointScore[score], not maximizing_player)
            else:
                winrate = self.simulation(board, new_taboo_list, scoredif-self.pointScore[score], not maximizing_player)
            if madeMove in mistake_moves:
                board.key = board.newHashOfMove(madeMove[0], True)
            else:
                board.unmakeMove(madeMove[0], madeMove[1])
            
            # If it still was a mistake, either return mistake, if we played a mistake move,
            # or add the move to the mistake moves and try again
            if winrate == "mistake":
                if madeMove in mistake_moves:
                    return winrate
                else:
                    possible_moves.remove(madeMove)
                    mistake_moves.append(madeMove)
                    board.makeMistake(madeMove)
            else:
                # If no mistake happend, return the result
                return winrate
    
    def jsonTableToNormal(self, jsonNodeTable, board):
        """
        Changes the json version of the precomputed node table to our normal node table
        """
        nodeTable = {}
        test: str = "test"
        for key in jsonNodeTable:
            stringHashes = key.split("_")
            newKey = (int(stringHashes[0]), int(stringHashes[1]))
            values = jsonNodeTable[key]
            first = values[0]
            second = values[1]
            third = []
            for moveList, stringChildKey in values[2]:
                move = Move(moveList[0], moveList[1], moveList[2])
                cell = [cell for cell in board.cells if cell.i == move.i and cell.j == move.j][0]
                stringChildHashes = stringChildKey.split("_")
                newChildKey = (int(stringChildHashes[0]), int(stringChildHashes[1]))
                third.append(((move, cell), newChildKey))
            forth = values[3]
            nodeTable[newKey] = (first, second, third, forth)
        return nodeTable
    
    # The precomputed data (removed for report)
    keyGenerator3x3 = """[[[5598062112450236371, 6699601199392269683, 49582051120476967, 2612058752370969759, 7287963225579376587, 7166253979797854320, 1285126200471472033, 5445945328790098562, 468189412764752015, 8039483350219525265], [5169648247748409165, 6471722519103225552, 3220570274422435062, 5598456530671611749, 8406190718384894560, 4299346098277239946, 1299740408461006260, 3150189477714512819, 1002513733798771793, 8290428755742917814], [101494262029352660, 8475292263854340619, 8802216999103441887, 5871590560229076116, 2176787147440525702, 850989360670383002, 1117629821909829054, 2387863798075390614, 610371978574369487, 1987651816748518931], [2763239510470895572, 5606880711194405806, 3830858707579761472, 8133065710853976523, 1440999473530835323, 4870727240311285415, 3800696003017833713, 6346142944511623958, 4034113454094828084, 4760769073669425215], [2932264163309951280, 1916514403845559112, 791948040513252150, 5557556837580685319, 2061865836224311177, 9000149167014452953, 1324314049149949068, 2096422064247448630, 1852342233231857489, 6939937541137616277], [7708049004119447527, 378215252990646102, 4913802340285809145, 7462117277061354014, 1700119173353553159, 6866791171216959243, 6450724141919527680, 3008227055076535342, 7940175752780965471, 8598905740327353742], [8206473607676120314, 8521261029445066816, 8556754126857861746, 864110542110008586, 5123764913187807041, 3214312913354434749, 2159645901310015684, 6391586477779823298, 8318019498723242078, 582269191121458798], [6427185327593684786, 368037312862391385, 6220150827356217666, 3400334553055578609, 1822528401466730962, 4658298799949027798, 1414144718300475302, 7720332182368513901, 8057701467313202336, 5343814405407070553], [6358555078764754442, 8875210079105951611, 4032814685140516432, 3892112946329720055, 2772153976943645343, 9191415846098824312, 8207583171780575087, 2881587615186985755, 9133425189471417431, 1104851290192093249]], [[6370919489794349453, 5371831439828163408, 2411206426905567143, 6724258906814837549, 4296044544999912960, 4645038414459837605, 5698816788082507484, 3149645663911209029, 1344842101758279063, 1437670253268145616], [6437223164084107410, 9060440781384684133, 2177717207061100443, 100422473495207503, 1731870920498680218, 492903039916525846, 7638743143456599299, 6407672215226231175, 6976169925179026129, 5246745069055984209], [9017966994311425198, 7580043331958552733, 1707394732674008959, 8791438764478897379, 4931229489082070734, 3898662316724442028, 3338298556123361277, 2168590739559812114, 4629087572256649022, 4504937347667823233], [1226290758479115770, 4394004544122032689, 2260517394861597906, 1791053419260865091, 1652721588042208569, 5063751392934352238, 8828526235040957793, 467751238314896076, 8446006742536281223, 515668057336332520], [9114656929711349298, 6142212367191150446, 7606566751442564857, 3183702573392491966, 3458782844213532881, 7327877287493317490, 5678630966297405840, 4686893634030323552, 3738492357531099182, 1067309194469333066], [6456873207640088911, 7854169429923587814, 5580763589311246295, 1839437451382578140, 8211043729542178007, 4823436652632836920, 7132296417205428981, 361774045041591473, 7383186076682397057, 6328893777160409401], [3249503837680803024, 1513354199860077526, 1355318202597824543, 5703512621344869575, 2491648330844662710, 7473569421225675005, 3367552722176675980, 7509377259233874259, 4912245566465689957, 1477106521546754704], [2586640844864651999, 1770767357885276457, 774645237717784324, 2982812806770724569, 1361352569488709481, 2853502935591364740, 5425485526200476143, 1276206731036687593, 8875120377582249949, 65004626036406969], [1522792106577960410, 6739307675611925144, 3424196804014473952, 1160513656776529179, 356258798403324898, 4745195652443914513, 6912658368887000244, 1561814175197713845, 1404613856286388452, 1763851373043998864]], [[7085629338180125937, 4781296996516154356, 3428076941067924296, 8914159473406084569, 394475915714598142, 7757676194961899119, 4055363415533856095, 4110244971419697406, 407828682771306847, 8245571858508872375], [8794348510296742749, 4893705703149816247, 3045059969595368593, 6230113547084566151, 1815365858691080190, 2433853617314064987, 8808975619568750905, 5270723110704962557, 1516399948882097105, 8384529070167744479], [704173079053396766, 2090142298348758221, 323441036617414542, 8771374420492119520, 1044710513634564350, 6846288960076454087, 1115493950508010754, 4388895236103927798, 8575905381992089760, 6314612274100053782], [2699418672906922320, 726681385416613464, 1415563380651596638, 1874952736232010768, 9080622509259201111, 8437567153880811442, 2690717641030176275, 4614886274070252675, 5300823794273693837, 3990206146380299406], [7686070942119767347, 1585798979819069403, 1111200699431993735, 1977690066219084106, 5183776680044485249, 5248218504013157376, 8342679663822517353, 5223260111974079254, 3599402408680245075, 8558154230944273802], [4192887065576819275, 4316224147137493684, 3089229275655193744, 5751173991718536837, 1857158006629824298, 5306948129886470388, 8095330449596922121, 6620299229655260008, 2729309952814473087, 583603580178970862], [8132862297448075295, 8099776859175924721, 1626765604824707703, 4485975121204289712, 8438625807635461112, 4552449924868086457, 4965085124209786094, 1638423590726896096, 8913121393054100047, 6734885792363973479], [2298960537336683256, 2979268691447652792, 8596607566610629753, 111542837129267162, 5636002479234566901, 2818446318577705046, 5846917309888670235, 5482927940567410086, 1573015131478468743, 3759691251877946813], [1440231621110603400, 6158683138668774337, 8249842873599203600, 4363342945704618297, 8807911731444361241, 138825856180611334, 4806583475989431341, 3863807211590299846, 1273775054554031605, 3782183718737152488]], [[6328787514482503863, 8228792742979022134, 1605445860806433967, 4832342085657451296, 5684827540564901130, 2582811022376559152, 1757228822532687054, 4900429843875520032, 6620579614102884034, 5697141700265400146], [5068721398102207907, 6773224172672728116, 3334304336258624599, 1146286500857147761, 6787479606848383349, 8031528642215021363, 3925765254773465543, 5111584310608486722, 8661817012887006931, 935834308945991406], [6250620283774796387, 6159891230382058386, 4194434593003434812, 2729466720717581311, 6199602219497812698, 9039165938205983679, 76512861398439934, 2145409356791506357, 4803632728653905036, 6939536793092436038], [5939317510680160843, 895458676051363789, 6738469590049050178, 2554664147842389379, 2908524865238752727, 5338456557996268864, 6387823045300233299, 8827767117900124934, 3358475311477999806, 1007025280894185591], [6661310175044368129, 8963757083498266040, 5502572690152606407, 1900567341971045293, 4527462786998386200, 7659352251768452923, 6738843153726631676, 745132916228806912, 892655253601379613, 346078595346620063], [7976678717592315720, 1835683189147215989, 9007866002896597075, 3259154813000751938, 5028722088206997933, 6694082543788037045, 1959352651050683268, 6587043664527804752, 2357815481269162692, 5105140856071391464], [1462821339467054887, 6356817966054707531, 1244037669969161430, 3494233439613436104, 8685729759949337822, 8048076025601698622, 8942058796495007460, 8351049969594254295, 9160808855274655732, 7942543242652583007], [8085537515176219447, 1594558784189031625, 8605257372086641100, 2843206901118362375, 6718108884734424881, 771288771741428712, 2085963316030494086, 8231310982879078325, 1718127292251316937, 3984400310125979396], [1050169882570334608, 4537519824962626372, 5798067683919155560, 4202022157348116569, 8388126560893065612, 5201435287262889449, 7043759917149310071, 5079757697711663193, 5975653379284224760, 6261935376358695638]], [[6246556103728736452, 3821746424209826162, 4176413103037381495, 760004427156909409, 2597548448754353761, 8167460231969823757, 765615526254090455, 280266209507701003, 4680717443145539271, 6883616306356396385], [8423345193186314445, 107084741665556566, 6298868973476670626, 715515214390555601, 656363731017106557, 3358873866083812992, 5516386492201166298, 1557326010727650499, 7870457567040996862, 4796185511401613170], [1881917326707396091, 1368531068023000054, 8173117793960340334, 3220047256313913845, 5857289556092558159, 1605933135253161451, 7958766246630711317, 264346574058318138, 3096913555228312863, 6695346135511561457], [4729955433881772837, 8521863432956340276, 5729905149445248846, 9054087730775153202, 6192736499922221285, 3808917255298017179, 6707157194930511312, 3234408954819144106, 3179819987451908396, 5435024166672106794], [8377301279008015637, 3552296750893174068, 408303245898844833, 8922944386792717199, 7755574384462729091, 3325347094428537881, 256562874254542030, 5470730604947167437, 973254692282136463, 7244689326082224459], [3402037515973499905, 930930225269470272, 2278421394614610754, 5830991853791891317, 5196178528459208870, 3189689288718929546, 496262180826824343, 548216450458729421, 1247057563456985282, 5539944517272465881], [1982732391352522252, 8322195034920984109, 6410171085477609117, 2407605837819125430, 2164404009282538524, 6711954302697930714, 4827828008542628575, 362528990975561981, 8931024170908340660, 9112655885539799448], [4501195275013465092, 2928356790213134199, 9172484353101799455, 5433528735015986647, 4125848390553042132, 210114903206129528, 3185385412503650639, 8618672619849786683, 4333265642577083497, 4283363208228866766], [2722779521259682198, 7022130340404786312, 1528844197015197249, 8103192767700553007, 2676231202689665526, 5915156005628935647, 4507861591910377694, 6256083890947657220, 5556869647754838757, 3959140665954615605]], [[9214495573708567232, 1093439263156518512, 351839759886112776, 7131428071501403153, 2146497830740578089, 8719471292722602934, 5372754376500014102, 8680522823467897875, 3051256056973256935, 8857694687009761300], [732005055528997863, 3424855970207124436, 7075761385944065379, 8213087469132851699, 7727530768554082944, 8303353827270989699, 1409805985966845860, 6397737786644996486, 8117227488337764771, 3055688753223371525], [3406581880080234775, 7646366143746327034, 2618281545920160665, 5634641173756893211, 1845350221662679514, 8995687712386804623, 1366144261864800178, 5761938682020141740, 8132604154028639791, 2377671908597808282], [6524036275253275685, 1865223690240276224, 3175012271493124565, 8078993119291056723, 3682765918367785308, 292376651325452590, 1822404869273164901, 4299007875972106378, 4303364063620784969, 605506723722156924], [6746144456857509228, 216804843586523304, 3283144624340317932, 4952810424670956914, 7808274495894094427, 4291892157944027252, 8000441985953259383, 4041722230745284988, 4752911576881023360, 6880011395674946952], [2571456685120906878, 7575641445200677583, 6786232699111937919, 9029555789687615671, 2834284410339022662, 5424407464546139111, 3997594497593336333, 5132666249096681910, 8139274845510300854, 9150235974729661025], [1939263489877481006, 6134778956347606247, 5144414393151972196, 5158004211328233244, 1959861401951760945, 2192660431709670255, 1503830882687024115, 1041813836008818937, 339938467148735904, 2495685158673540719], [6503445957139836392, 9213538840729970524, 787941614489782063, 4842800148558454346, 6658355500170068843, 8108832712991000758, 3711510480892853897, 8163016132390534568, 6288246102461085345, 4657563227383561133], [55748392785868590, 593044685976426361, 6906226478731652102, 167749025946991059, 782443774251986788, 1146263461201009259, 7629081590933042886, 9039846483737794629, 7607073524364626291, 4966321123595427769]], [[6440568226737838166, 9188580748286223820, 7889622619604550400, 694875081684785691, 6482579612528259482, 2193549176919106518, 6528674533842715699, 4720308764732921449, 5982771762424699313, 8779273299357261257], [8403364420042588593, 7624655940980484268, 3458493464526170432, 3744452012117967176, 7817880572061756872, 5016943337585240477, 2227058391401578595, 1565884990844895485, 6483518012616605448, 4707527914320899991], [1846614615566833050, 4903273357251452086, 3429907920903626263, 1390906077216370783, 6535416372698535875, 6829894924231177544, 1649932465889647170, 2560094386735440671, 8278963894357767818, 897769683478396188], [9034342255483799066, 7188453337529521362, 5098503798302149182, 2585431090378719734, 28825839530715845, 6581396877110719242, 4807771621035859247, 3796286970744171361, 1405782495241279631, 4800396854784625306], [2480769714603776208, 3086587509716994514, 6816471818698102610, 6476800883163888072, 1881732720716871929, 1529175403514642212, 4145903432694665528, 1092522965343037403, 8776222580777225279, 8778082073592989950], [4267912786324016054, 3367547445900904675, 1576283178067542761, 5830795281646021133, 1234564612115061278, 3628900726403003731, 7857838166651164727, 8089704218087606917, 4389543519882227279, 1448666620648753371], [1274418889672365922, 4357373750395132330, 2586302791569283566, 8616808378771115381, 1032051606537135051, 1789273488376155164, 506869422347769898, 1082874437934298188, 8741758111677283292, 7147381158262733362], [526968196304338433, 4325402689085411690, 7328441176833344501, 2067437406152105799, 8294336540415484614, 2818909181926377448, 4895651560319918563, 147594035378584527, 6274548195103389551, 6235346157313433007], [2754178607260625354, 4836627262667405439, 2651434908715816145, 6912049123191811900, 5655329312134147617, 9010842670344414580, 3309837673738407686, 2678891293912762152, 1050633569303836945, 877928971943367286]], [[190292023527704529, 5246106259460657699, 9098429171059494422, 1026025160928003260, 1913119180755338606, 6387481632941040679, 2190951328456317871, 8079776314256580839, 5106998107191768760, 8901710746473183144], [4126110813791352631, 2750141358476650356, 4059685666238565740, 3398304717164756235, 1031810028145473812, 2265401964770848884, 1872250285804840995, 4816410645641373033, 8855417600037592021, 5046590355845749668], [8878501191039400115, 4632479073035969413, 4265381526691651845, 2717697138978965093, 5362723658389340337, 3764025024532278438, 2881109263215813043, 4718375321296662855, 4365946795740798474, 4359356240277779324], [6462475654934668463, 4792652058891183192, 4628147813926156132, 4837261881420337125, 5222562156341942065, 5048997370547085682, 577899669759873575, 8636614727338191605, 2512701897302688521, 2730814318015854611], [5538769010008507450, 6546575741475087308, 3906668226551989025, 8873026597494006359, 6907657712530348842, 8743794116026290275, 1314813368369018086, 2527683457672933102, 2852090509988955276, 67542272510447499], [8536391731373662425, 1508660098364990094, 5786363665646421227, 7941172635013984895, 2418541840318576391, 4772159763670313693, 3152260135331290163, 8539071918337097376, 8782066008359099846, 9089664326132404127], [3962605195109372626, 2353077860644126665, 761229059643043143, 8216528622912289063, 9012577282564962801, 6184094350561872808, 6493341104470397676, 3227885390165963076, 4864985370329966998, 7343084242475170184], [2408561164560732765, 5874664101833362583, 5825638519259528609, 4677432500276832964, 6390419109207333926, 5927425942552861918, 2921256840163736643, 4463406998542446843, 2229792733372014801, 3543242036933032926], [1011780977398272727, 7578452048320071401, 369715950954630062, 2496900617512865672, 4324948863908063062, 601028546191437921, 2056212725195626398, 4003054499342739102, 5425637846983097123, 3416137281121470811]], [[1351522775290129449, 2926027931828873798, 2783453441422556049, 2611202910047841207, 3808343418700293666, 8735696611912437675, 4275858318511596897, 3975567939600722286, 5631609031963744363, 2119424602703073431], [1211312546199429434, 4145137943190069146, 8700118711283093117, 2924117340862379253, 9052127478530131424, 2643711056721857085, 5592075290290739480, 3780187295541162089, 6543593649273469356, 1480999724711794757], [1708851229252579589, 763466079567398085, 7687134652264261825, 6169851376474365187, 4002039444128036773, 6225780534786152931, 5889320062709711260, 283416578374542819, 6527187903046878092, 8436422404711411487], [7856472878084942463, 3993342942859786487, 1673161524517414347, 6381366361651613197, 4931864987625741366, 1224203500667343291, 4307802670198972099, 4296225641867083902, 3648230055305684593, 4608985216498854312], [9166396750982748129, 1552016611136069145, 3054780462162025453, 3091352389088027913, 1679614268314736609, 5083604265030166702, 8673371617693294373, 2580602093027261752, 1807838168582874338, 8900240571422363671], [4128321735743962831, 8013517822161346000, 807569116584951503, 7692081322988815546, 2173325279998879381, 7967766432625545141, 210465078542434940, 5216827443829533430, 3101343478251411345, 3329091673485266825], [2044600340016631742, 1951152920982248430, 3201958227661872965, 5353898720481657606, 1412173665319719171, 3205491773706413962, 4858880181373091639, 7325887778266086019, 8298622435340725155, 3104383509264701334], [1669227946999422401, 5227205526906547373, 3473917468670465915, 9134343926709191083, 2521237645623671559, 1079215723538996867, 7729947317512788986, 6374092506259528509, 8759295248907650838, 6814873087126459401], [6154129192074098206, 5650507204826696963, 2346942003879249817, 4987074413478843961, 4218292226595524296, 3240855075569097136, 5020023082752627380, 152575995591496402, 5269805441551256095, 7465506460441677850]]]"""
    moveKeyGenerator3x3 = """[[[5672774555816957269, 1108337322528428201, 5653772025884572708, 8668524546350717131, 1929894420557740477, 3881686652757186061, 5278039788427331836, 5425666893781978164, 1134143484042145579, 4522667313380931156], [3780366326022652653, 4097092126996026368, 2289002517651361206, 7812752454150450060, 6741938524002983420, 5599443473112598537, 5768401852186188614, 4157194370554482888, 6850149475798327997, 9106222828751872940], [4663734173073483370, 7651857854766534655, 6564059230564052259, 445105511560567106, 675154433197411205, 2952029244602798968, 2732166637104532767, 9168070670794591468, 7870547366865524446, 6519463030028135401], [3407166809676249627, 6921796146602978337, 2527293217214631466, 5929917124741691918, 5104411734581187241, 6346730705439460213, 5751577083896656692, 4394293465594417770, 1656301390818267959, 2259223738565482382], [5864105183416370559, 3489854565178236156, 1394722015290148236, 53986659294642534, 7784835920824930560, 7598175097098767946, 1148900219035022883, 2437061008280075111, 4352410709985322771, 2344208428353845271], [4074032818795691049, 4986508213053278111, 5529513246579576492, 5343621274891693806, 5155047866229023983, 2350054822575596081, 1063922109753290218, 6000090821155425398, 458760535703979260, 5577464039054094511], [6885585124801832955, 8545308658654659, 7644376495213336505, 7801090315910923128, 1380027964313886171, 1669099003628268147, 719756106268610270, 4400200024095921950, 4991018219430734584, 1078771051946629433], [4844395783637151755, 391542079411755944, 4769534573598213514, 3020444074795785502, 1245884898977853972, 1530435384842096399, 4688806307441401985, 7315735029316815030, 4462980548478768991, 6472336759812905561], [6573318128214746984, 1029600485764984929, 4921219233585838073, 7114519129311774066, 4662455047364819658, 805756019812562067, 7136478808194012655, 7023296083218946217, 8271283618636797722, 1557383227125424310]], [[3636421792294410324, 5814089715908988367, 7987145405021606631, 3132241594141521902, 3173130813094625368, 4573499359980846561, 1420436791717933309, 1825049937877810416, 4271518955275644612, 1456522336961866705], [5084281249978875036, 5348613471617087106, 6697287281128832008, 6540163471535479393, 2220376152284483435, 6059340618823497911, 6188390472423321345, 7140098610844768909, 7527870299388619989, 3226493946607892989], [7045285295979222043, 5850364926716472751, 4040645677918383489, 675708044745533187, 266555633564404242, 5066976395550865937, 2501603299095593987, 7730002099307650039, 7029836627506268406, 3257798843079341814], [5962726244362459045, 7517765691280227650, 5349905614084887786, 8459870583560424512, 3281392256342110441, 1731708265573411274, 7838933407311981020, 7285704007488761501, 5273969873174751999, 5402617859721311825], [954883609352530104, 8046466189339227722, 6908128524730599902, 1898706088991801050, 898050157899498851, 1836106024451898359, 7008002420412619193, 6731539204732560586, 7371704958387982351, 8188776501326580424], [1478137502675317787, 6953592262254408837, 1640011362637719738, 3034116445301321395, 4602491813646410118, 3464176127382281800, 3235203866007202063, 2570395073183323986, 7091044211823420140, 6092904169834424732], [2581689898352145819, 8655612076474244646, 5718707288505765618, 6991371275201101848, 8472640755129228372, 2572672225437633751, 5917133104235266416, 2604832626196184332, 6247942208162801989, 5832273463842941029], [7167347090338443609, 6823471102212601523, 4406067676852587860, 8842524924209529490, 4093494177333359995, 2444571392711590039, 4629087713900046564, 7163937466491893160, 1653216473451114547, 8709998566724471634], [6991445482905399895, 3814428130362080666, 5025151940295042425, 5670057140398561901, 8856744007850916451, 5302739766248122808, 8937821415953486212, 5523394240720768056, 5714967502914705817, 6074298376013707304]], [[4280970793407660834, 9092161550997876750, 1920720393230590475, 1047865954748906926, 8224605571722820495, 5708682500600021611, 247052643494551421, 844381219277508778, 7780286442183405595, 2050526406254166023], [5235425841621882734, 2635662797904993719, 3170604392025675248, 6062789576856948185, 1756468513970920075, 6643530950170024950, 8142052440029117348, 5945601262487332246, 1615839915215219693, 2919347190221657419], [1836825749039638395, 5824470745412816815, 6219344703709505121, 4988547635519376889, 8493130763281019108, 70758345790294131, 7224684062626046435, 5720282265271560927, 1708297750727856983, 6385504886673926463], [4447890524249693209, 3488002351975174541, 7796018971901679348, 1425963401470436175, 3650287532175862583, 5829461015333779895, 5960864189203209582, 7137426475564534925, 7249433161892874880, 7763367398559866228], [3623515547908647858, 6437071542737393937, 2663980779893688578, 2493380546494517492, 8116907465040092533, 4068701965936661077, 615436757300725915, 5265098049517126556, 877487114494538331, 1600423309029084906], [7684321708676716714, 5342602530394218431, 7737566423198121617, 664015314732841191, 7033844560713380803, 5707386210010994396, 8399168739601921675, 2107993234874985329, 931199722557886673, 4146743491415621204], [6082024039682727215, 3639963595517864533, 6452601365426727866, 1480528966674411621, 7769143176722956078, 381970539216912283, 4175851698297291250, 4486757205940975376, 5772679260980234290, 941939216527602289], [1606481508188161098, 7882268098811347846, 3817179327747144451, 2365754804183121835, 4807183382391443529, 1029607857570914332, 3400685395941427676, 5763159490988816313, 6480788071986106476, 2120010330263850897], [372268856757914749, 2476096961716512825, 9073127309556017122, 1842979617399707700, 2003154639602991477, 5463733501303764199, 921280414814363574, 4249285337131424639, 6377157037818633081, 8348813362331935042]], [[9213808773343037113, 4720573749442608372, 6422514571599203360, 3169384753392614244, 336201961943573679, 2461813325727097761, 1053312173694147502, 1874340679691835006, 3147256908089736949, 7624125126545183522], [5942126537165721129, 5285076712674532427, 4708903282469141866, 101603519101625829, 8456099948270254089, 806597554532907369, 2988852037495055213, 314063621685361396, 8045865569865088849, 2691085053718732221], [1387274877018026130, 7501834726678756393, 3680960350633316238, 7196940774714754822, 5153951393515460961, 1798842733484170410, 1756421091799422259, 4787380107299858720, 1822982397743906402, 8084534691777815726], [4569229976647239973, 2512191765854035339, 2937131350964155504, 8947509959103638460, 3147401581832982800, 2058963405832109084, 3479815921115502867, 2422787696899818558, 4121551922346459349, 3666777240478610572], [624740544002048448, 4148830089932382741, 4210446051286740156, 3719056454985620319, 6640514236129438126, 7114305218250720439, 7862850211650379621, 7626013277230445035, 3861303665795281830, 8118175098472478344], [5458923608301629567, 584909626064447152, 5311047191041584183, 2402931712822985659, 8082842555521395884, 7118257396206571930, 942787614013765309, 5747773911865731212, 2224914437352440723, 1517174161353371874], [7823251083786639738, 718013511425199877, 8262563233072775984, 6975722551672731722, 6951707910389490536, 8252216222929801269, 5977360236918013390, 3086720071977117915, 2714674251689723935, 2447454742140910264], [3695847361776924511, 2648333195924519078, 3769764189275974509, 3147504026618977894, 4384980294654681812, 203518425287208197, 9109457741076056793, 7021077512437351056, 911791930455754328, 5591341864661449313], [8352869851472886295, 1413159341272536776, 7618029451270964556, 3543747739319783737, 5364344036041303856, 4944775536034672665, 4029541252824172359, 2305820324715634566, 5849277762151631512, 1404277583914152394]], [[6790559344158315918, 5220672441317600204, 5964157513971292646, 2188783948538267501, 6058584061661809147, 7130392158729796715, 7068421865518421649, 3193267637355043904, 4504275599028079365, 2022188891460220399], [5742827350034332034, 5539092302412754216, 8717347920155401007, 8562683249326013590, 710882830884866424, 8227057576505126499, 1453202077298131175, 2484379341536094488, 8569681299520377611, 4419319451324883199], [1574708722777688391, 3632366126377292124, 3274133047338160475, 3889412142957646123, 8993559000718341276, 3626828745806753083, 2265955194864023016, 2917453795528795337, 7754143142493926049, 5074702765969270643], [3502659473665025644, 3513859743750048909, 6121724174843560680, 7298149528887367919, 9083424922039216204, 1817056157791282432, 7244448899474785241, 6951267686629492519, 4419152868915216229, 3440837257566317476], [973742665036327703, 7717999360995833835, 1514610897148167361, 8646193472592808888, 4263746143631896873, 5975498180415911765, 3088666436713242469, 9047351100758087057, 3565459050362273481, 532626680208105626], [3982327061271784325, 8105284781599611611, 7955353003531980297, 2596461025937018451, 8552593422508013404, 3270621059423057969, 3176601072918364843, 6555420977160438481, 831178917246391936, 7284789424510044177], [5382727517499121589, 8402132316097900880, 4082891176476788849, 9075114139914712201, 2374143484007345539, 4634695620942961549, 3001274009952743735, 637946142054689459, 3774431076843536172, 3707958143901653201], [8363318771908794331, 7876660128398755514, 4830026256899979377, 3849226939926323551, 6487714149443595382, 3858933750925303187, 5820928921512026252, 929490058801331739, 6121131503531198573, 7058018311104435030], [6199474335375692656, 5376175231805570533, 3259242156465526143, 5056395116402911935, 2591366227584937836, 5592227434633637282, 8980619705868590615, 7011004123596829231, 1811976828782605595, 195472799117470773]], [[2794784644967232660, 1476749577293846731, 8040145846765595009, 1098904725025812392, 7826773775220301003, 6222616718032209828, 5046489397496081018, 7484517746856209771, 1718084387310343243, 3994339228167417995], [7502937137685865542, 6491070811276859772, 8737649057912502842, 5411697664595709702, 6541922074328091270, 7312467596450025066, 737641265595757134, 7399516368912394058, 2676363754500831613, 8309380836203430553], [399585811948778055, 1089384022709418452, 2144081064898882359, 6862149094346330169, 5709993401843898007, 5306466083060498831, 8826674324350102888, 380216505555465348, 609096456087145017, 4405063932916887155], [804319692164350998, 408108241615879230, 3756277360790660379, 9060942686890865538, 1672499003387505962, 6864604301953581808, 9154311453534292899, 6477023460980248652, 8806755565498300819, 308348820932219080], [1495322551347681958, 6426905374848510725, 11870114051412660, 5969670604479471378, 8540851883915718014, 7422400402321863211, 5619560603253644179, 2066552636037902944, 2406462476490794060, 2752578904094787387], [2677462116106248199, 8351689923700306806, 3028994905847699933, 1524202424634538898, 234077069321215647, 1779952149573224252, 1222243057782023152, 7393729195262764935, 581234882762058615, 4452037467331981610], [505696012716361539, 4912787201077927463, 7638941547688190948, 8691113688809511071, 1264645463303578588, 5674750482777070505, 5463110843289382104, 1245998646719572454, 7481994326004711521, 4222351796633223887], [2869725366663288731, 6659095908296336213, 4541146690486940367, 2231017995922236800, 8455194345132511188, 2328612951184699707, 8574054884030669606, 2343840684207352950, 6427855658275366126, 8116425442712231510], [3097251924535180167, 3241630153056643742, 3179626788125262241, 6438383903345027391, 4298461064181489590, 8023203697100057325, 3231933611983527671, 4691749391906623611, 6617500952463373572, 2882839197987481104]], [[6370584944211618166, 5090569685893703877, 511391325155664760, 2679423539532630897, 6478678143725030690, 1135055224450931547, 3271197543118822894, 8249659780196198650, 9039163526424884015, 5072531917298478400], [2105821329275884503, 1205266024940716105, 4483150051071859574, 4449938454727136130, 1519004255879733042, 1065631802213107744, 5857440280087015695, 8494608334652241487, 7812306631824545835, 3660327685444439139], [2279522155470235773, 6537654089519437415, 1268968572938355933, 9206745190016148492, 5761639357667610671, 1845720328651941993, 178283569567486985, 7185498338808334960, 5452285649122301915, 7372807042959526700], [3424800561851856407, 8548854101108149900, 6306922764809263657, 5813887664566915208, 1759826811877807504, 7632555828578982038, 8581528674341646445, 5903668748578301239, 7180659480087908225, 1941888819824085856], [7175001855521636070, 903163478439385022, 6178744342854067357, 8536562922673318049, 8591744614753863064, 6193172495540892457, 4670328641060763716, 5133030499929914672, 6699623509896832437, 8074782326566193027], [8406630877963124464, 852512264991471962, 1074497385440935337, 1715096466977304928, 62960546391519206, 4645498845683309838, 4703187143516494883, 5984621092053706072, 7652405894439440554, 6716652640735793437], [9018848674983116951, 1343393945763793406, 2204799265305738112, 501125742570522126, 2798729444191276326, 3686486039637286245, 8535264046813754370, 6565577627504525397, 3902124258302561875, 6305601194215604999], [2558545207411318934, 4433219546893508986, 5063679179297044655, 6331729765834769788, 6043270226782813903, 7036231208273436001, 4751012296667463496, 4033375500395204674, 3831100463851508857, 3609392396926591202], [1013522953384049447, 7064410174623208927, 993998044547146782, 6454730331197101570, 8013339891588490979, 2458265889512794986, 6597597659179609227, 2425033675477406602, 1922770365126645534, 1573926261426578876]], [[1564181319546563214, 8986992388412367432, 345232421183120519, 4492454900807756254, 7785039422817867048, 3907090368417740761, 3278586351514111019, 3808089228594421557, 4573177157703086360, 5309857085642633201], [3967577299387582959, 1550896747921171328, 8909030869627501742, 8379392502181781095, 831162728322447394, 6163359391950969737, 5381711903303040239, 5967191546161209384, 6050153887874792251, 2438461696260867301], [8203862727199200124, 5454879337245861537, 1696046785914418918, 442886813438273091, 4296576648187741782, 3015660183288829059, 7225954883056622834, 3703595692047229444, 6135154452812984387, 7869106002607738279], [6100836931548819486, 8532540230950243019, 764851608024505406, 5360101205301687257, 3950416597421199788, 5978347484620971239, 8002191536177668325, 6315382707836705623, 5764085810533318828, 8113464259060955804], [424916922129974969, 4744071139228255869, 423121355511141501, 2787935922485943756, 3541379341341973182, 7214963977185783752, 5601157601403452799, 3975429417433289080, 2267338253065382653, 4212430428917548854], [2334855454101139011, 4191089092737349145, 4584113946228018124, 5240722799489816375, 337890291590505145, 3816602015065301698, 5552398827628739286, 621535062267617631, 6930696264704536357, 8489257157902125337], [6785344843778493658, 5025436292025699051, 209169636061342944, 215210694005230971, 3155550657964141981, 2132506839619565783, 1995130567997089501, 7127607858903071005, 7152945772608118354, 5044538327779547733], [8380899274062973446, 5634926562459788910, 2758968746995130516, 2086400222766052602, 3972547727697233157, 544550783542803743, 9089139441742020691, 5069083683103344465, 6848533394941679509, 2246412815095982067], [4150291776395978511, 1930285003105159665, 7912329213776471422, 8378514410130396228, 854377404668188502, 8283929228534380491, 9086147316852370923, 5822974102555695013, 3189729757957278328, 6751394410324404138]], [[8843410489712256818, 237895237944063367, 6061039647015053565, 700695827262504632, 6756372770251496220, 348876499303040273, 3369065617276350728, 5957608280754714329, 671161185193638671, 7276614354075472870], [8071269327912195994, 6091599364257501563, 7901362547441129608, 8786217871072935966, 4137153956923974711, 1392851606769583596, 7298949028791996738, 3492706774967578558, 7054524651663465840, 4118098675424426470], [7203152628951209110, 1001494121375840252, 6093864534887548921, 1598963969508965813, 7073266884803339591, 621699847438651366, 8753355549549675426, 6111207424630726468, 3865261684858721623, 3942265204265975552], [852566201281832422, 547724739947404630, 3625472723174239125, 1495152032907823288, 8727096680656767649, 7331840902017855770, 2181919096449906496, 4848811634086165893, 4476295975781184637, 5910798844663583927], [3333656908636303167, 2018974472707268045, 1591127670412830469, 7134328583442526129, 3835928551283406990, 7117988472305484195, 5315037107945659032, 6757674359900350842, 6776628862241891108, 8937860826853403749], [3100227848114343769, 5543731912249190206, 6178711572651532194, 6812594344158584368, 6266540453822769512, 8068786649304534165, 4623551766859597709, 2140296787336112323, 6057386767742567939, 7098677885849831472], [1812232619306626577, 3048278538085780081, 2056827382781200754, 4799508938135855692, 8020175283798572835, 7238760868797258236, 6548299532461091834, 8462417946341614956, 123433677233169272, 8211560208981267066], [6113798455913399374, 6507552884816144905, 5001646824931736534, 1838289100389705218, 8207424963252727784, 846793502598998290, 61714182856270016, 1364766215218324893, 8624159180604559812, 1714587895683638735], [4418070702871507915, 8675289799578424678, 7828973679109799515, 2851521234027163577, 4711374416830410686, 2312801056151444964, 4217144244495393298, 3069446815344898233, 3689418270995692824, 815956306954945109]]]"""
    nodeTable3x3 = """{"0_531058869563243793": [-45, 100, [[[2, 4, 7], "5269805441551256095_3131893643416458895"]], [7, 12, -17, 13, -13, 5, -11, -11, 17, 15, -17, 13, -2, -19, -13, -19, 9, 3, 15, -13, 7, -17, -15, -3, -11, 3, 25, -7, -9, 19, 5, -13, 7, 7, 1, 21, 5, -1, 11, -11, 11, 11, -21, 15, -13, -19, -1, -7, 21, -9, 5, -15, -1, 9, 13, -17, -9, -1, -1, 7, 3, -1, -17, -7, -9, 7, -5, -17, 14, -21, -19, 7, 1, 5, -3, 5, -11, -9, -5, -5, -11, 17, -17, -2, 11, -15, -11, 8, -21, 3, -12, 25, 13, -25, -19, 27, -9, 9, 13, -15]], "5269805441551256095_3131893643416458895": [44, 99, [[[2, 3, 4], "7750640113087142817_6757722587812849935"], [[6, 4, 3], "8280585167573247588_2584550622067339606"], [[5, 8, 9], "8934620002904788756_2114448958627755400"], [[4, 5, 5], "4226475280360026724_7332081668598636777"], [[2, 0, 9], "8269417522932339965_8913501923988234637"], [[2, 6, 6], "2960026561196967235_1334744053385681173"], [[0, 0, 5], "1419571381909714727_7534264408703736472"], [[0, 1, 4], "8157336330383767921_1724223976857163250"], [[2, 8, 8], "5284327578598648305_3862972055463443453"], [[7, 0, 6], "6380252533107822942_3090356834831014080"], [[4, 6, 5], "6150326700016591987_3687105304896734382"], [[6, 4, 3], "4799691955043704546_3431510410253099996"], [[2, 6, 9], "4728191801151126340_9022214579256696530"], [[7, 7, 1], "903700312484686787_2326136507767516668"], [[0, 1, 5], "2881877143641017773_8078120101706858551"], [[6, 7, 2], "8885313582797056270_6478553357177021212"], [[0, 1, 4], "8141037817464693848_548293520098526301"], [[8, 6, 4], "480120664513034098_5053534572880991258"], [[7, 7, 6], "1161512606387670629_5220072106473654395"], [[2, 6, 3], "1926239033757426384_781235675375887319"], [[4, 0, 9], "4536467999930530973_6616542251937374751"], [[1, 8, 6], "8903981026638026180_5022246078374535974"], [[4, 5, 7], "4506126811328482520_1841354473438315667"], [[2, 3, 8], "8439098696049908338_971498872802247648"], [[2, 4, 5], "8768450894725855389_3632068867113198102"], [[3, 5, 1], "1714314424076810837_990184219136035176"], [[6, 3, 5], "2710805082082045568_1855739084699289483"], [[2, 5, 5], "2085131039660405251_4857758951846308774"], [[5, 0, 9], "5420189325556213369_1117952943817790781"], [[1, 5, 1], "6128734273341957565_8643123156912468581"], [[3, 8, 9], "6935987174100823218_1780826940110875215"], [[1, 1, 3], "1287614086640679087_6966635032657568521"], [[5, 7, 8], "7503594250565198563_8456955279984356659"], [[7, 4, 3], "3676340005264837375_6838285725359682752"], [[2, 6, 5], "7099918811907771990_8180391059692693401"], [[1, 1, 2], "6053299121096407709_7423755763599619992"]], [12, -17, 13, -13, 5, -11, -11, 17, 15, -17, 13, -2, -19, -13, -19, 9, 3, 15, -13, 7, -17, -15, -3, -11, 3, 25, -7, -9, 19, 5, -13, 7, 7, 1, 21, 5, -1, 11, -11, 11, 11, -21, 15, -13, -19, -1, -7, 21, -9, 5, -15, -1, 9, 13, -17, -9, -1, -1, 7, 3, -1, -17, -7, -9, 7, -5, -17, 14, -21, -19, 7, 1, 5, -3, 5, -11, -9, -5, -5, -11, 17, -17, -2, 11, -15, -11, 8, -21, 3, -12, 25, 13, -25, -19, 27, -9, 9, 13, -15]], "7750640113087142817_6757722587812849935": [0, 3, [[[6, 4, 4], "2965093251887458093_3445446496645943005"], [[1, 1, 5], "6665066127978418498_1990454501244158650"]], [-17, -1, -19]], "8280585167573247588_2584550622067339606": [-2, 2, [[[2, 1, 5], "5201036654564081223_6712804945362795676"]], [13, 9]], "8934620002904788756_2114448958627755400": [-1, 3, [[[5, 5, 8], "5378723273023286018_712976157454156993"], [[6, 8, 8], "7222076816019757947_6888641245216735189"]], [-13, 11, -11]], "4226475280360026724_7332081668598636777": [-2, 2, [[[7, 0, 2], "425925328556109508_2458996260979137871"]], [5, 7]], "8269417522932339965_8913501923988234637": [0, 3, [[[2, 0, 2], "4252277847615446114_686241585926295501"], [[8, 0, 5], "8392337347047883344_5648911909967290483"]], [-11, -11, -5]], "2960026561196967235_1334744053385681173": [0, 4, [[[4, 7, 9], "1502491102480620704_1929420567354243203"], [[1, 4, 1], "8236099398661983533_2290653013349806545"], [[0, 0, 3], "3052307754622074062_8584428301681935071"]], [-11, -15, -11, -9]], "1419571381909714727_7534264408703736472": [-2, 2, [[[1, 7, 8], "6430588518804109818_3946917274363648248"]], [17, 1]], "8157336330383767921_1724223976857163250": [-1, 3, [[[7, 3, 3], "2768734100184203454_5115616320875537891"], [[3, 4, 7], "423347754696603464_6339248252465626151"]], [15, -17, -19]], "5284327578598648305_3862972055463443453": [0, 3, [[[7, 0, 1], "3471905214057188345_1934170358373886090"], [[4, 6, 9], "2694091823551103627_2902640927579869088"]], [-17, -21, -5]], "6380252533107822942_3090356834831014080": [-2, 3, [[[1, 6, 4], "6342592710710790286_7960533992648313737"], [[4, 2, 8], "1224986823742095862_2930472052964918554"]], [13, -1, 8]], "6150326700016591987_3687105304896734382": [0, 3, [[[8, 0, 9], "2168516144290933251_1719460753627230851"], [[1, 0, 2], "3954862927981008628_5604251768443249643"]], [-2, -9, -3]], "4799691955043704546_3431510410253099996": [-2, 3, [[[4, 4, 7], "6278769523633163138_2024151101631622283"], [[8, 0, 8], "802474085497112891_3730502515687635488"]], [-19, 15, 13]], "4728191801151126340_9022214579256696530": [-1, 3, [[[1, 2, 3], "4499897135445841055_6026578822077056338"], [[2, 6, 8], "1355777424466643391_7610337054837382521"]], [-13, -1, 17]], "903700312484686787_2326136507767516668": [-2, 3, [[[6, 5, 7], "1178590112469717939_8936226980862740002"], [[7, 3, 8], "4018257150694957885_3790904353860461313"]], [-19, 13, 27]], "2881877143641017773_8078120101706858551": [-1, 3, [[[2, 6, 3], "52745608644880653_6556975893501134664"], [[3, 4, 5], "6807017018023732962_5977304733320017908"]], [9, -1, -21]], "8885313582797056270_6478553357177021212": [-2, 3, [[[4, 8, 8], "4204413359891550798_2531607765602762445"], [[4, 2, 4], "6822545866824595057_2001805937935158946"]], [3, -21, 13]], "8141037817464693848_548293520098526301": [-1, 3, [[[2, 5, 1], "8822734853630932605_8871758763418529370"], [[8, 8, 8], "8103270317554318728_7977215116085644274"]], [15, -1, -15]], "480120664513034098_5053534572880991258": [0, 3, [[[6, 4, 2], "1033098468448566136_4182393001651053908"], [[4, 2, 6], "7944214221593380146_6970879985232304786"]], [-13, -19, -11]], "1161512606387670629_5220072106473654395": [-1, 3, [[[5, 0, 9], "1789789899066693205_326466394604514892"], [[0, 5, 1], "8669257538159375777_7545216215289136208"]], [7, -5, -25]], "1926239033757426384_781235675375887319": [0, 3, [[[4, 3, 3], "7019101831582449387_6714121211075301039"], [[8, 1, 9], "3755277326078976581_88473713408927130"]], [-17, -17, -2]], "4536467999930530973_6616542251937374751": [-1, 3, [[[2, 3, 1], "1756945565109524533_7330609603297334572"], [[1, 2, 7], "7233343182201634577_1702305124701610571"]], [-15, 5, -12]], "8903981026638026180_5022246078374535974": [-2, 3, [[[2, 7, 6], "8833617906907274417_2659454829568204961"], [[6, 4, 1], "4856546443489821314_8705299530733447105"]], [-3, 5, 25]], "4506126811328482520_1841354473438315667": [-2, 3, [[[5, 0, 7], "9094313961223600166_9159637015768437602"], [[3, 4, 6], "8122509874449109965_1281042778362339620"]], [-11, 11, 11]], "8439098696049908338_971498872802247648": [-2, 3, [[[5, 1, 5], "749965543872237281_5048875126612279654"], [[5, 6, 8], "4911712287925626036_5982803637500682586"]], [3, -17, 3]], "8768450894725855389_3632068867113198102": [-2, 2, [[[3, 5, 4], "5231934140968404571_7790090744789663916"]], [25, 7]], "1714314424076810837_990184219136035176": [0, 3, [[[8, 2, 4], "3703073576872725585_273206215602084811"], [[6, 0, 1], "2258385242586250185_4114449370335740363"]], [-7, -13, -9]], "2710805082082045568_1855739084699289483": [-1, 2, [[[2, 1, 1], "6093917129342565875_219523446132471191"]], [-9, 21]], "2085131039660405251_4857758951846308774": [-2, 2, [[[5, 8, 6], "625721316935194361_1510800015238213782"]], [19, 7]], "5420189325556213369_1117952943817790781": [-2, 3, [[[2, 7, 1], "9037388747340795425_1245206260769732786"], [[2, 2, 4], "5271021867629540523_6756489837231907870"]], [5, -7, 9]], "6128734273341957565_8643123156912468581": [0, 3, [[[3, 7, 3], "576007776059432589_6272589884997471832"], [[0, 3, 7], "5598705675526542528_2876673990745741460"]], [-13, -7, -17]], "6935987174100823218_1780826940110875215": [-2, 2, [[[1, 5, 9], "1399470576900767731_7534131159800458023"]], [7, 14]], "1287614086640679087_6966635032657568521": [-1, 2, [[[3, 0, 1], "1948039612489332936_1184477531160394719"]], [7, -9]], "7503594250565198563_8456955279984356659": [-2, 2, [[[6, 2, 2], "7353182042665747501_3900950739299123068"]], [1, 5]], "3676340005264837375_6838285725359682752": [-2, 2, [[[8, 8, 5], "2918251773737330152_6720547354270798013"]], [21, 3]], "1756945565109524533_7330609603297334572": [1, 1, [], [5]], "7099918811907771990_8180391059692693401": [0, 3, [[[4, 3, 5], "1540061858104110044_199131409994221530"], [[8, 6, 7], "4472286515232249796_8243861802216802533"]], [-1, -9, -15]], "6053299121096407709_7423755763599619992": [-2, 2, [[[2, 4, 1], "7285378098043745844_3350210924691034026"]], [11, 5]], "4252277847615446114_686241585926295501": [0, 1, [], [-11]], "9094313961223600166_9159637015768437602": [1, 1, [], [11]], "5378723273023286018_712976157454156993": [1, 1, [], [11]], "3471905214057188345_1934170358373886090": [0, 1, [], [-21]], "6278769523633163138_2024151101631622283": [1, 1, [], [15]], "3703073576872725585_273206215602084811": [0, 1, [], [-13]], "1033098468448566136_4182393001651053908": [0, 1, [], [-19]], "4499897135445841055_6026578822077056338": [0, 1, [], [-1]], "576007776059432589_6272589884997471832": [0, 1, [], [-7]], "6093917129342565875_219523446132471191": [1, 1, [], [21]], "2168516144290933251_1719460753627230851": [0, 1, [], [-9]], "8833617906907274417_2659454829568204961": [1, 1, [], [5]], "1502491102480620704_1929420567354243203": [0, 1, [], [-15]], "2965093251887458093_3445446496645943005": [0, 1, [], [-1]], "5201036654564081223_6712804945362795676": [1, 1, [], [9]], "1178590112469717939_8936226980862740002": [1, 1, [], [13]], "7019101831582449387_6714121211075301039": [0, 1, [], [-17]], "1540061858104110044_199131409994221530": [0, 1, [], [-9]], "8822734853630932605_8871758763418529370": [0, 1, [], [-1]], "6342592710710790286_7960533992648313737": [0, 1, [], [-1]], "625721316935194361_1510800015238213782": [1, 1, [], [7]], "2918251773737330152_6720547354270798013": [1, 1, [], [3]], "52745608644880653_6556975893501134664": [0, 1, [], [-1]], "749965543872237281_5048875126612279654": [0, 1, [], [-17]], "9037388747340795425_1245206260769732786": [0, 1, [], [-7]], "1948039612489332936_1184477531160394719": [0, 1, [], [-9]], "425925328556109508_2458996260979137871": [1, 1, [], [7]], "1789789899066693205_326466394604514892": [0, 1, [], [-5]], "2768734100184203454_5115616320875537891": [0, 1, [], [-17]], "1399470576900767731_7534131159800458023": [1, 1, [], [14]], "4204413359891550798_2531607765602762445": [0, 1, [], [-21]], "6665066127978418498_1990454501244158650": [0, 1, [], [-19]], "5231934140968404571_7790090744789663916": [1, 1, [], [7]], "6430588518804109818_3946917274363648248": [1, 1, [], [1]], "7285378098043745844_3350210924691034026": [1, 1, [], [5]], "3954862927981008628_5604251768443249643": [0, 1, [], [-3]], "7353182042665747501_3900950739299123068": [1, 1, [], [5]], "8236099398661983533_2290653013349806545": [0, 1, [], [-11]], "2258385242586250185_4114449370335740363": [0, 1, [], [-9]], "2694091823551103627_2902640927579869088": [0, 1, [], [-5]], "8392337347047883344_5648911909967290483": [0, 1, [], [-5]], "7944214221593380146_6970879985232304786": [0, 1, [], [-11]], "1355777424466643391_7610337054837382521": [1, 1, [], [17]], "5598705675526542528_2876673990745741460": [0, 1, [], [-17]], "3755277326078976581_88473713408927130": [0, 1, [], [-2]], "8122509874449109965_1281042778362339620": [1, 1, [], [11]], "4472286515232249796_8243861802216802533": [0, 1, [], [-15]], "7222076816019757947_6888641245216735189": [0, 1, [], [-11]], "1224986823742095862_2930472052964918554": [1, 1, [], [8]], "6807017018023732962_5977304733320017908": [0, 1, [], [-21]], "4911712287925626036_5982803637500682586": [1, 1, [], [3]], "7233343182201634577_1702305124701610571": [0, 1, [], [-12]], "4856546443489821314_8705299530733447105": [1, 1, [], [25]], "802474085497112891_3730502515687635488": [1, 1, [], [13]], "8669257538159375777_7545216215289136208": [0, 1, [], [-25]], "423347754696603464_6339248252465626151": [0, 1, [], [-19]], "4018257150694957885_3790904353860461313": [1, 1, [], [27]], "3052307754622074062_8584428301681935071": [0, 1, [], [-9]], "5271021867629540523_6756489837231907870": [1, 1, [], [9]], "6822545866824595057_2001805937935158946": [1, 1, [], [13]], "8103270317554318728_7977215116085644274": [0, 1, [], [-15]]}"""
    
    
    
    