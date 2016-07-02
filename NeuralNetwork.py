# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 09:31:11 2015
@author: Sundip R. Desai
Neural Network 
--------------
Rev 1:
Leverages matrix/vector operations for
feedforward and backpropagation routines which
substantially improves computational time
Written ad-hoc for 1 hidden layer
Rev 2 to-do: 
1. Gradient checking
2. Regularization
3. Cross-Entropy formulation in backprop routine
4. Possible momentum parameter
5. Confusion Matrix Output
6. Option to use random seed for weight initialization
"""
import numpy as np
# Libraries

# Load data into arrays
# data slicing is ad-hoc, need to modularize
# somehow
def loadData(name):
   data = np.loadtxt(name)
   inputs = data[:,1:10] 
   targets = data[:,-1:]   
   return data, inputs, targets
   
# Scale data from [0,1] for sigmoid activation
def scaleData(data):
    for x in range(data.shape[1]):
        xmin=min(data[:,x])
        xmax=max(data[:,x])
        data[:,x]=(data[:,x]-xmin)/(xmax-xmin)
    return data

# Convert raw output to class designation <i.e., either 1,2 or 3>
def output2Class(c):
    a = np.zeros((c.shape[0],1))
    for x in range(c.shape[0]):
        m = max(c[x,:])
        for y in range(c.shape[1]):
            if c[x,y] == m: a[x] = y+1
    return a            

# Convert class to raw output designation <i.e., 2 = [0,1,0]>
def class2Output(data):         
    a = np.zeros((len(data), 3))
    for x in range(len(data)):
        a[x,data[x]-1]=1
    return a
    
# Use this classification error function
# for binary targets 
def binaryClassError(o,t):
    n=0
    o=np.round(o)
    for x in range(len(o)):
        if o[x] == t[x]: n+=1
    e=float(n)/len(o)*100
    return e       
    
def classificationError(o,t):
    n=0
    for x in range(o.shape[0]):
        m=max(o[x,:])
        mm=[i for i,j in enumerate(o[x,:]) if j==m]
        if (mm == (t[x]-1)):n+=1
    e=float(n)/len(o)*100  #compute error at end of loop
    
    return e
    
# Activation Function -- Sigmoid
def activation(z):
    return 1.0/(1.0+np.exp(-z))

# Derivative of activation
def activation_prime(z):
    return activation(z)*(1-activation(z)) 
    
# Initialize Weights    
def initWeights_andBiases(sizes):
    # Insert an argument of sizes
    #i.e., [10 4 3] -> 10 input neurons, 4 hidden, 3 output neurons

    # x = input count
    # y = hidden count
    # z = output count

   w=[np.random.randn(x, y) for x, y in zip(sizes[:-1], sizes[1:])]
   b=[np.random.rand(x,1) for x in sizes[1:]]
   
   return w, b
    
# Composes a vector of bias values for vectorized
# computation    
def fetchBiasVec(b,n):
    
    a=np.ones((n,1))
    b1=np.dot(a,b[0].reshape(1,len(b[0])))
    b2=np.dot(a,b[1].reshape(1,len(b[1])))

    return b1, b2
    
def feedForward(x,y,b1,b2):
    #x - weights
    #y - input matrix
    #z - bias vector    

  net= [np.zeros((n, m)) for m, n in zip([x[0].shape[1],x[1].shape[1]],[y.shape[0],y.shape[0]])]
  out= [np.zeros((n, m)) for m, n in zip([x[0].shape[1],x[1].shape[1]],[y.shape[0],y.shape[0]])]
            
  net[0] = np.add(np.dot(y,x[0]),b1)
  out[0] = activation(net[0])    
  net[1] = np.add(np.dot(out[0],x[1]),b2)  
  out[1] = activation(net[1])  
     
  return  net, out
   
def networkError(targets,out, crossEntropy=False):
    # User has option to use cross entropy as the error function
    # or mean squared error
    # Default it mean squared error
    
    if (crossEntropy):
        crossent=np.multiply(targets,np.log(out))+np.multiply((1-targets), np.log(1-out))
        error = -np.sum(np.sum(crossent,1))/len(targets)  
    else:    
        # Compute mean square error
        error = np.sum(np.subtract(targets,out[1])**2)/(targets.shape[0]*targets.shape[1])

    return error


def backProp(i,t,b,b1,b2,o,n,w,eta, crossEntropy=False):
    #i - inputs
    #t - targets
    #b - bias (just a vector of ones)
    #o - out
    #n - net 
    #w - weight matrix
    #eta - learning rate

    if (crossEntropy==False):
        # Output->Hidden Layer
        error_vector = np.subtract(t,o[1])
        delta = np.multiply(error_vector, activation_prime(n[1]))
        weights_delta = np.dot(np.transpose(o[0]), delta) 
        db1=np.sum(np.multiply(delta,b2),0)
        # Compute new weights and biases for hidden->output layer
        ww1= np.add(w[1], eta/o[1].shape[0]*weights_delta)
        b_1=np.add(b[1], eta/o[1].shape[0]*db1.reshape(db1.shape[0],1))
    
        # Hidden->Input Layer
        delta2 = np.dot(delta,np.transpose(w[1]))
        n = np.multiply(delta2, activation_prime(n[0]))
        weights_delta2 = np.dot(np.transpose(i),n)
        db2=np.sum(np.multiply(delta2,b1),0)
        # Compute new weights and biases for input->hidden layer
        ww2 = np.add(w[0],eta/o[0].shape[0]*weights_delta2)
        b_2=np.add(b[0], eta/o[1].shape[0]*db2.reshape(db2.shape[0],1))
    
    # Capture updated weights/biases in array 
    w=[ww2, ww1]
    b=[b_2,b_1]
    
return w,b
