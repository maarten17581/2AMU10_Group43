import random
import time
import math
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
from team43_A3_NN.region import Region
from typing import List
import numpy as np

class NeuralNetwork:
  """Neural network class"""
  def __init__(self, layer_structure, file_name):
    """Create layer structure"""
    if file_name == None:
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
    else:
      with open(file_name, "r") as file:
        parameter_text = file.readlines()
        parameters = []
        for line in parameter_text:
          stripped_line = line.strip()
          parameters.append(stripped_line)
        self.L = int(parameters[0])
        self.layer_structure = []
        for i in range(1,self.L+1):
          self.layer_structure.append(int(parameters[i]))
        self.weights = []
        self.biases = []
        self.weightVelocity = []
        self.biasVelocity = []
        index_count = self.L+1
        for i in range(1,self.L):
          weight_matrix = []
          for j in range(self.layer_structure[i]):
            weight_row = []
            for k in range(self.layer_structure[i-1]):
              weight_row.append(float(parameters[index_count]))
              index_count += 1
            weight_matrix.append(weight_row)
          self.weights.append(np.array(weight_matrix))
          self.weightVelocity.append(np.zeros((self.layer_structure[i], self.layer_structure[i-1])))
        for i in range(1,self.L):
          bias_vector = []
          for j in range(self.layer_structure[i]):
            bias_vector.append(float(parameters[index_count]))
            index_count += 1
          self.biases.append(np.array(bias_vector))
          self.biasVelocity.append(np.zeros(self.layer_structure[i]))
        
    
  def call(self, x):
    """Forward pass"""
    current = x
    for i in range(0,self.L-1):
      current = np.dot(self.weights[i], current) + self.biases[i]
      if i != self.L-2:
        current = np.maximum(0, current)
    return current
  
  def total_forward(self, x):
    """Forward pass returning all layer values"""
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
    """Back propogation"""
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
    for e in range(epochs):
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
      validationLossSum = 0
      for j in range(len(validation)):
        call = self.call(validation[j][0])
        validationLossSum += self.loss(validation[j][1], call)
      validationLossAvg = validationLossSum/len(validation)
      learning_rate *= learning_multiplier
      if validationLossAvg < lowestValidationLoss:
        lowestValidationLoss = validationLossAvg
        bestWeights = self.weights
        bestBiases = self.biases
    self.weights = bestWeights
    self.biases = bestBiases
    finalValidationLossSum = 0
    for j in range(len(validation)):
      call = self.call(validation[j][0])
      finalValidationLossSum += self.loss(validation[j][1], call)
    finalValidationLossAvg = finalValidationLossSum/len(validation)
    print(f"Final validation loss after training: {finalValidationLossAvg}")
      
  def print_to_file(self, path):
    """Print weights to file"""
    with open(path, 'w') as file:
      file.write("%s\n" % self.L)
      for layer in self.layer_structure:
        file.write("%s\n" % layer)
      for weight_layer in self.weights:
        for weight_row in weight_layer:
          for weight in weight_row:
            file.write("%s\n" % weight)
      for bias_layer in self.biases:
        for bias in bias_layer:
            file.write("%s\n" % bias)
    print('Updated file')