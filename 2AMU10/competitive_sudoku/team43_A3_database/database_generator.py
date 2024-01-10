from datetime import datetime
import random
import re
import sys
sys.path.append('C:\\Users\\20202991\\Dropbox\\My PC (S20202991)\\Documents\\GitHub\\Confuente_Website\\2AMU10_Group43\\2AMU10\\competitive_sudoku')
from sudokuai import SudokuAI
from regionsudokuboard import RegionSudokuBoard
from competitive_sudoku.sudoku import GameState, SudokuBoard, TabooMove, Move
from competitive_sudoku.sudoku import load_sudoku
from competitive_sudoku.execute import solve_sudoku
import math
import numpy as np
import copy
import itertools
import json

def main():
    # Load the sudoku and the ai
    startboard = load_sudoku("boards\empty-3x3.txt")
    game_state = GameState(startboard, startboard, [], [], [0, 0])
    board = RegionSudokuBoard(game_state)
    ai = SudokuAI()
    
    # Create the table
    nodeTable = {}
    time = datetime.now()
    lasttime = time
    count = 100
    # Iterate mcts for {count} amount of times
    for i in range(count):
      print(f"start {i}")
      ai.mcts(nodeTable, board, [], 0, True)
      print(f"end {i}")
      # Print some time estimation
      timenow = datetime.now()
      duration = timenow-time
      lastduration = timenow-lasttime
      lasttime = timenow
      avgDuration = (duration.total_seconds())/(i+1)
      expDuration = (count-i-1)*avgDuration
      print(f'{i}/{count} expected time: {int(expDuration//3600)}:{int((expDuration//60)%60)}:{int((expDuration)%60)}')
      print(f'time last iteration: {int(lastduration.total_seconds()//3600)}:{int((lastduration.total_seconds()//60)%60)}:{int((lastduration.total_seconds())%60)}')
    for key in nodeTable:
      print(f"{key} {nodeTable[key][0]} {nodeTable[key][1]} {nodeTable[key][3]}")
      
    # Make the table json printable
    printable_node_table = {}
    for key in nodeTable:
      stringKey = str(key[0])+"_"+str(key[1])
      first = nodeTable[key][0]
      second = nodeTable[key][1]
      third = []
      for move, childHash in nodeTable[key][2]:
        third.append([[move[0].i, move[0].j, move[0].value], str(childHash[0])+"_"+str(childHash[1])])
      forth = nodeTable[key][3]
      printable_node_table[stringKey] = [first, second, third, forth]
    
    # Store the table and the hash tables
    json.dump(board.keyGenerator, open("keyGenerator.txt",'w'))
    json.dump(board.moveKeyGenerator, open("moveKeyGenerator.txt",'w'))
    json.dump(printable_node_table, open("nodeTable.txt",'w'))
    
    # Check if we can get them back
    new_nodeTable = json.load(open("nodeTable.txt"))
    
    for key in new_nodeTable:
      print(f"{key} {new_nodeTable[key][0]} {new_nodeTable[key][1]} {new_nodeTable[key][2]} {new_nodeTable[key][3]}")
    
if __name__ == '__main__':
    main()