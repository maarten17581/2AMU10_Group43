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

class NeuralNetwork:
  """Dense neural network class."""
  def __init__(self, layer_structure):
    """Creation of the layers"""
    self.L = len(layer_structure)
    self.layer_structure = layer_structure
    self.biases = []
    self.weights = []
    self.biasVelocity = []
    self.weightVelocity = []
    for i in range(1,self.L):
      self.biases.append(np.random.rand(layer_structure[i]))
      self.weights.append(np.random.rand(layer_structure[i], layer_structure[i-1]))
      self.biasVelocity.append(np.zeros(layer_structure[i]))
      self.weightVelocity.append(np.zeros((layer_structure[i], layer_structure[i-1])))
    
  def call(self, x):
    """Forward pass."""
    current = x
    for i in range(0,self.L-1):
      current = np.dot(self.weights[i], current) + self.biases[i]
      if i != self.L-2:
        current = np.maximum(0, current)
    return current
  
  def total_forward(self, x):
    """Forward pass returning all layer values."""
    layers = [x]
    current = x
    for i in range(0,self.L-1):
      current = np.dot(self.weights[i], current) + self.biases[i]
      if i != self.L-2:
        current = np.maximum(0, current)
        layers.append(current)
    layers.append(current)
    return layers
  
  def back_prop(self, y, layers):
    """Back propogation."""
    dLoss = 2*(layers[-1]-y)
    dLayerWeight = [np.dot(np.transpose([dLoss]), [layers[-2]])]
    dLayerBias = [dLoss]
    dA = np.dot([dLoss], self.weights[-1])
    for i in range(len(layers)-3, -1, -1):
      dZ = np.where(dA > 0, 1, 0)
      dLayerWeight.append(np.dot(np.transpose(np.multiply(dA, dZ)), [layers[i]]))
      dLayerBias.append(np.multiply(dA, dZ)[0])
      dA = np.dot(np.multiply(dA, dZ), self.weights[i])
    return list(reversed(dLayerWeight)), list(reversed(dLayerBias))
    
  def loss(self, y, y_hat):
    """Loss function"""
    loss = np.sum((y-y_hat)**2)
    return loss
  
  def train(self, train, validation, batch_size=100, epochs=1000, start_learning_rate=0.01, end_learning_rate=0.0001, momentum=0.9):
    """Training the Neural Net"""
    random.shuffle(train)
    lowestValidationLoss = math.inf
    bestWeights = []
    bestBiases = []
    learning_rate = start_learning_rate
    learning_multiplier = (end_learning_rate/start_learning_rate)**(1/epochs)
    batchCount = 0
    noImprovementCount = 0
    for e in range(epochs):
      maxChange = 0
      for i in range(len(train)//batch_size):
        if batchCount >= len(train):
          random.shuffle(train)
          batchCount = 0
        batchCount += batch_size
        dWeightsSum = None
        dBiasesSum = None
        for j in range(batch_size):
          layers = self.total_forward(train[(i*batch_size+j)%len(train)][0])
          dLayerWeights, dLayerBiases = self.back_prop(train[(i*batch_size+j)%len(train)][1], layers)
          if dWeightsSum == None:
            dWeightsSum = dLayerWeights
            dBiasesSum = dLayerBiases
          else:
            dWeightsSum = [dWeightsSum[k]+dLayerWeights[k] for k in range(len(dWeightsSum))]
            dBiasesSum = [dBiasesSum[k]+dLayerBiases[k] for k in range(len(dBiasesSum))]
        dWeightsAvg = [dWeightsSum[j]/batch_size for j in range(len(dWeightsSum))]
        dBiasesAvg = [dBiasesSum[j]/batch_size for j in range(len(dBiasesSum))]
        for j in range(len(self.weights)):
          self.weightVelocity[j] = momentum*self.weightVelocity[j] - learning_rate*dWeightsAvg[j]
          self.biasVelocity[j] = momentum*self.biasVelocity[j] - learning_rate*dBiasesAvg[j]
          self.weights[j] += self.weightVelocity[j]
          self.biases[j] += self.biasVelocity[j]
          maxChange = max([maxChange, np.amax(learning_rate*dWeightsAvg[j]), np.amax(learning_rate*dBiasesAvg[j])])
      validationLossSum = 0
      for j in range(len(validation)):
        call = self.call(validation[j][0])
        validationLossSum += self.loss(validation[j][1], call)
      validationLossAvg = validationLossSum/len(validation)
      trainingLossSum = 0
      for j in range(len(train)):
        call = self.call(train[j][0])
        trainingLossSum += self.loss(train[j][1], call)
      trainingLossAvg = trainingLossSum/len(train)
      learning_rate *= learning_multiplier
      print(f"Validation loss at epoch {e}: {validationLossAvg}")
      print(f"Training loss at epoch {e}: {trainingLossAvg}")
      print(f"Max parameter change at epoch {e}: {maxChange}")
      if validationLossAvg < lowestValidationLoss:
        lowestValidationLoss = validationLossAvg
        bestWeights = copy.deepcopy(self.weights)
        bestBiases = copy.deepcopy(self.biases)
        noImprovementCount = 0
      else:
        noImprovementCount += 1
      if noImprovementCount >= epochs//50:
        print(f"Premature convergence at epoch {e}")
        break
    self.weights = bestWeights
    self.biases = bestBiases
    finalValidationLossSum = 0
    for j in range(len(validation)):
      call = self.call(validation[j][0])
      finalValidationLossSum += self.loss(validation[j][1], call)
    finalValidationLossAvg = finalValidationLossSum/len(validation)
    print(f"Final validation loss after training: {finalValidationLossAvg}")
      
  def print_to_file(self, path):
    """
    Prints the weights to a file
    """
    with open(path, 'w') as fp:
      fp.write("%s\n" % self.L)
      for layer in self.layer_structure:
        # write each item on a new line
        fp.write("%s\n" % layer)
      for weight_layer in self.weights:
        for weight_row in weight_layer:
          for weight in weight_row:
            fp.write("%s\n" % weight)
      for bias_layer in self.biases:
        for bias in bias_layer:
            fp.write("%s\n" % bias)
    print('Updated file')

def is_taboo(board: RegionSudokuBoard, move: Move):
  """
  Determines if any move is a mistake move in the current position
  """
  solve_sudoku_path = 'bin\\solve_sudoku.exe'
  board_text = str(board)
  options = f'--move "{(move.i*board.N+move.j)} {move.value}"'
  output = solve_sudoku(solve_sudoku_path, board_text, options)
  if 'has no solution' in output:
    return True, 0
  if 'The score is' in output:
    match = re.search(r'The score is ([-\d]+)', output)
    if match:
      player_score = int(match.group(1))
      return False, player_score
    else:
      raise RuntimeError(f'Unexpected output of sudoku solver: "{output}".')

def create_data(board: RegionSudokuBoard, game_state: GameState, main_nn: NeuralNetwork, count, alpha, gamma):
  """
  Creates data for the neural network
  """
  data = []
  time = datetime.now()
  for _ in range(count):
    start_depth = random.randint(0,board.emptyCount-1)
    startBoard = RegionSudokuBoard(game_state)
    scoredif = 0
    maximizing = True
    taboo_moves = []
    for __ in range(start_depth):
      possibleMoves, mistakeMoves = startBoard.get_moves(taboo_moves)
      moves = [move for move in possibleMoves + mistakeMoves if TabooMove(move[0].i, move[0].j, move[0].value) not in taboo_moves]
      move, cell = random.choice(moves)
      isTabooMove, score = is_taboo(startBoard, move)
      if isTabooMove:
        taboo_moves.append(TabooMove(move.i, move.j, move.value))
        board.key ^= board.keyGenerator[move.i][move.j][move.value]
      else:
        startBoard.makeMove(move, cell)
        if maximizing: scoredif += score
        else: scoredif -= score
      maximizing = not maximizing
    if not maximizing:
      scoredif = -scoredif
    input = input_vector(startBoard, taboo_moves)
    nn_output = main_nn.call(input)
    output = [-100]*(board.N**3)
    afterMoveOutputs = []
    possibleMoves, mistakeMoves = startBoard.get_moves(taboo_moves)
    moves = [move for move in possibleMoves + mistakeMoves if TabooMove(move[0].i, move[0].j, move[0].value) not in taboo_moves]
    for move, cell in moves:
      move_index = move.i*(board.N**2)+move.j*board.N+move.value-1
      isTabooMove, score = is_taboo(startBoard, move)
      if isTabooMove:
        new_taboo_moves = taboo_moves
        new_taboo_moves.append(TabooMove(move.i, move.j, move.value))
        nextInput = input_vector(startBoard, new_taboo_moves)
        nextOutput = main_nn.call(nextInput)
        minOutput = min([nextOutput[move[0].i*(board.N**2)+move[0].j*board.N+move[0].value-1] for move in moves if TabooMove(move[0].i, move[0].j, move[0].value) not in new_taboo_moves])
        output[move_index] = nn_output[move_index] + alpha*(score + gamma*minOutput - nn_output[move_index])
        nextMoveIndices = [move[0].i*(board.N**2)+move[0].j*board.N+move[0].value-1 for move in moves if TabooMove(move[0].i, move[0].j, move[0].value) not in new_taboo_moves]
        afterMoveOutputs.append((move_index, scoredif, score, nextInput, nextMoveIndices))
      else:
        startBoard.makeMove(move, cell)
        newPossibleMoves, newMistakeMoves = startBoard.get_moves(taboo_moves)
        newMoves = [move for move in newPossibleMoves + newMistakeMoves if TabooMove(move[0].i, move[0].j, move[0].value) not in taboo_moves]
        nextInput = input_vector(startBoard, taboo_moves)
        startBoard.unmakeMove(move, cell)
        minOutput = 0
        if len(newMoves) == 0:
          if scoredif+score > 0: minOutput = 100
          else: minOutput = -100
        else:
          nextOutput = main_nn.call(nextInput)
          minOutput = min([nextOutput[move[0].i*(board.N**2)+move[0].j*board.N+move[0].value-1] for move in newMoves if TabooMove(move[0].i, move[0].j, move[0].value) not in taboo_moves])
        output[move_index] = nn_output[move_index] + alpha*(score + gamma*minOutput - nn_output[move_index])
        nextMoveIndices = [move[0].i*(board.N**2)+move[0].j*board.N+move[0].value-1 for move in newMoves if TabooMove(move[0].i, move[0].j, move[0].value) not in taboo_moves]
        afterMoveOutputs.append((move_index, scoredif, score, nextInput, nextMoveIndices))
    data.append((input, output, afterMoveOutputs))
    timenow = datetime.now()
    duration = timenow-time
    avgDuration = (duration.total_seconds())/(_+1)
    expDuration = (count-_-1)*avgDuration
    print(f'{len(data)}/{count} expected time: {int(expDuration//3600)}:{int((expDuration//60)%60)}:{int((expDuration)%60)}')
  return data

def update_data(datapoints, neural_net: NeuralNetwork, N, alpha, gamma):
  """
  Updates each datapoint so the output is the next step in temporal difference learning
  """
  data = []
  for datapoint in datapoints:
    input = datapoint[0]
    output = [-100]*(N**3)
    previousOutput = datapoint[1]
    nn_output = neural_net.call(input)
    afterMoveInputs = datapoint[2]
    for moveIndex, scoredif, score, nextInput, nextMoveIndices in afterMoveInputs:
      minOutput = 0
      if len(nextMoveIndices) == 0:
        if scoredif+score > 0: minOutput = 100
        else: minOutput = -100
      else:
        nextOutput = neural_net.call(nextInput)
        minOutput = min(nextOutput[index] for index in nextMoveIndices)
      output[moveIndex] = nn_output[moveIndex] + alpha*(score + gamma*minOutput - nn_output[moveIndex])
    data.append((input, output, afterMoveInputs))

def input_vector(board: RegionSudokuBoard, taboo_moves):
  """
  Creates an input vector for the neural network based on the current board position and the taboo moves
  """
  input = [0]*(2*(board.N**3))
  for cell in board.cells:
    if cell.value != 0:
      input[cell.i*(board.N**2)+cell.j*board.N+cell.value-1] = 1
  possibleMoves, mistakeMoves = board.get_moves(taboo_moves)
  moves = [move for move in possibleMoves + mistakeMoves if TabooMove(move[0].i, move[0].j, move[0].value) not in taboo_moves]
  for move, cell in moves:
    input[board.N**3+move.i*(board.N**2)+move.j*board.N+move.value-1] = 1
  return input
    
def create_perm(n, m, N):
  """
  Creates all possible digit, row and column permutations given the size of the sudoku
  """
  digit_perm = list(itertools.permutations(range(N)))
  row_perm = []
  col_perm = []
  for possible_row_perm in list(itertools.permutations(range(N))):
    possible = True
    for i in range(n):
      box = possible_row_perm[i*m]//m
      for j in range(1,m):
        if possible_row_perm[i*m+j]//m != box:
          possible = False
          break
      if not possible:
        break
    if possible:
      row_perm.append(possible_row_perm)
  for possible_col_perm in list(itertools.permutations(range(N))):
    possible = True
    for i in range(m):
      box = possible_col_perm[i*n]//n
      for j in range(1,n):
        if possible_col_perm[i*n+j]//n != box:
          possible = False
          break
      if not possible:
        break
    if possible:
      col_perm.append(possible_col_perm)
  return digit_perm, row_perm, col_perm

def apply_perm(data_point, N, digit_perm, row_perm, col_perm, count):
  """
  Applies a permutation to the given datapoint
  """
  data = []
  for _ in range(count):
    this_digit_perm = random.choice(digit_perm)
    this_row_perm = random.choice(row_perm)
    this_col_perm = random.choice(col_perm)
    new_input = []
    new_output = []
    new_afterMoveInput = []
    for row in range(N):
      for col in range(N):
        for value in range(N):
          new_input.append(data_point[0][this_row_perm[row]*(N**2)+this_col_perm[col]*N+this_digit_perm[value]])
          new_output.append(data_point[1][this_row_perm[row]*(N**2)+this_col_perm[col]*N+this_digit_perm[value]])
    for row in range(N):
      for col in range(N):
        for value in range(N):
          new_input.append(data_point[0][N**3+this_row_perm[row]*(N**2)+this_col_perm[col]*N+this_digit_perm[value]])
    for moveIndex, scoredif, score, nextInput, nextMoveIndices in data_point[2]:
      new_moveIndex = this_row_perm[moveIndex//(N**2)]*(N**2)+this_col_perm[(moveIndex%(N**2))//N]*N+this_digit_perm[moveIndex%N]
      new_nextInput = []
      for row in range(N):
        for col in range(N):
          for value in range(N):
            new_nextInput.append(nextInput[this_row_perm[row]*(N**2)+this_col_perm[col]*N+this_digit_perm[value]])
      for row in range(N):
        for col in range(N):
          for value in range(N):
            new_nextInput.append(nextInput[N**3+this_row_perm[row]*(N**2)+this_col_perm[col]*N+this_digit_perm[value]])
      new_nextMoveIndices = []
      for nextMoveIndex in nextMoveIndices:
        new_nextMoveIndices.append(this_row_perm[nextMoveIndex//(N**2)]*(N**2)+this_col_perm[(nextMoveIndex%(N**2))//N]*N+this_digit_perm[nextMoveIndex%N])
      new_afterMoveInput.append((new_moveIndex, scoredif, score, new_nextInput, new_nextMoveIndices))
    
    data.append((new_input, new_output, new_afterMoveInput))
  return data

def main():
    # Get starting board
    startboard = load_sudoku("boards\empty-2x2.txt")
    
    # Make the gamestate
    game_state = GameState(startboard, startboard, [], [], [0, 0])
    
    # Get our board datastructure
    board = RegionSudokuBoard(game_state)
    
    # Make the neural net
    main_nn = NeuralNetwork([2*(board.N**3), board.N**5, board.N**3])
    
    # Create all the data pair
    data = create_data(board, game_state, main_nn, 1000, 0.1, 0.9)
    
    # Create all possible permutations for digit, row and column swaps
    digit_perm, row_perm, col_perm = create_perm(board.n, board.m, board.N)
    
    perm_data = []
    iterator = 0
    for data_point in data:
      # For each datapoint we apply a number of permutations and add it to the data
      perm_data.extend(apply_perm(data_point, board.N, digit_perm, row_perm, col_perm, 100))
      iterator += 1
    data.extend(perm_data)
    
    # Create a train and validation dataset
    train = data[(len(data)//10):]
    validation = data[:(len(data)//10)]
    print("training data created")
    
    # Train for this amount of iterations
    training_iterations = 1000
    for iter in range(training_iterations):
      
      # Train the neural net on this iteration of the data
      main_nn.train(train=train, validation=validation, start_learning_rate=0.00001, end_learning_rate=0.000001)
      print("iteration "+str(iter)+" trained")
      
      # Store the neural net
      main_nn.print_to_file("trained_weights.txt")
      
      # Update the data for the next output step
      update_data(data, main_nn, board.N, 0.1, 0.9)
      print("data updated")
    
if __name__ == '__main__':
    main()