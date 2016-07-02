# -*- coding: utf-8 -*-
"""
Executive for Neural Network Program
Author: 
Sundip R. Desai
Description: 
This script calls the relevant functions inside the
NeuralNetwork.py to generate a single hidden layer 
neural network for classification/pattern recognition 
purposes.
The user can generate a network with any number of inputs, hidden
neurons and outputs. Moreover, the user may use the mini-batch feature
to improve the network learning by using a subset of inputs and
learning weights accordingly respective to that subset. This is called
'batch learning' as opposed to 'online' learning where a single input
is fed-forward and weights adjust accordingly to that single input.
This script will evaluate the performance of a training set, validation set
(used to preclude overfitting) and test set.
"""

import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt

# Load in data
trainData, trainInputs, trainTargets = loadData('data_tumor_train.txt')
valData, valInputs, valTargets = loadData('data_tumor_val.txt') 
testData, testInputs, testTargets = loadData('data_tumor_test.txt')  

# Scale data
trainInputs=scaleData(trainInputs)
valInputs=scaleData(valInputs)
testInputs=scaleData(testInputs)

trainTargets=scaleData(trainTargets)
valTargets=scaleData(valTargets)
testTargets=scaleData(testTargets)

# Obtain class outputs
trainClassOutput=output2Class(trainTargets)
valClassOutput=output2Class(valTargets)
testClassOutput=output2Class(testTargets)


# Number of Inputs, Hidden Layer Neurons, Output Neurons
# This code works for 1 hidden layer only
numI = trainInputs.shape[1]
numH = 8
numO = trainTargets.shape[1]
numB = len(trainInputs) #trainInputs.shape[0] # data in batch
eta = .8  #learning rate
epochs = 100 #iterations
batchNum = 10 # number of mini batches

# Initialize weights/biases
# arg takes in sizes = [inputN, hiddenN1, hiddenN2,...hiddenNn, outputN]
weights, biases = initWeights_andBiases([numI,numH,numO])

# Vector initialization
mse=np.zeros(epochs)
valErr=np.zeros(epochs)
testErr=np.zeros(epochs)
mseVal=np.zeros(epochs)
mseTest=np.zeros(epochs)
cErr=np.zeros(epochs)

# Neural Network Learning Loop
for k in range(epochs):
    
    # Fetch a vector of permutated indices for next
    # training session, initialize indices to 
    # gather mini batches of data and initialize
    # total mean squared error term
    perm=np.random.permutation(len(trainInputs)) 
    index2=batchNum
    index1=0    
    mseTotal=0
    
    # Loop over all mini-batches
    for kk in range(len(trainInputs)/batchNum):
        
        # Gather mini batch of inputs and targets
        miniBatchInput=trainInputs[perm[index1:index2],:]
        miniBatchTarget=trainTargets[perm[index1:index2],:]
        
        # Increment indices
        index1+=batchNum
        index2+=batchNum
        
        # Compose bias vectors 
        bias1,bias2 = fetchBiasVec(biases, len(miniBatchInput)) 
    
        # Net, Out for each node and layer
        net, out = feedForward(weights, miniBatchInput, bias1, bias2)
    
        # Update weights/biases via backpropagation
        weights,biases = backProp(miniBatchInput, miniBatchTarget, biases, bias1, bias2, out, net, weights, eta)
        
    # Training Error
    bias1,bias2 = fetchBiasVec(biases, len(trainInputs)) 
    net, out = feedForward(weights, trainInputs, bias1, bias2)
    error = networkError(trainTargets, out)
    mse[k]=error
    
    # Validation Run
    # Compose bias vectors 
    bias1,bias2 = fetchBiasVec(biases, len(valInputs)) 
    netV, outV = feedForward(weights, valInputs, bias1, bias2)
    validation_error = networkError(valTargets, outV)
    valErr[k]=validation_error
    
    # Test Run
    bias1,bias2 = fetchBiasVec(biases, len(testInputs)) 
    netT, outT = feedForward(weights, testInputs, bias1, bias2)
    test_error = networkError(testTargets, outT)
    testErr[k]=test_error
    
    # Classification Error from Test Samples
    #class_error=classificationError(outT[1], testTargets)
    class_error=binaryClassError(outT[1], testTargets)
    cErr[k]=class_error
    
# Plot training/validation/test error history  
plt.plot(range(epochs), mse*100, label='Training Error')
plt.ylabel('Mean Squared Error')
plt.xlabel('Epochs')
plt.grid()  
plt.plot(valErr*100, label='Validation Error')
plt.plot(testErr*100, label='Test Error')
plt.legend()
plt.show()

# Print Classification Error
plt.plot(range(epochs), cErr)
plt.ylabel('Classification Error')
plt.xlabel('Epochs')
plt.grid()
plt.show()

# Print Statistics
successRate=(1-min(mse))*100
print "Success rate of Neural Network is: ", successRate

meanClassErr=np.mean(cErr)
print "Mean Classification Error is: ", meanClassErr
