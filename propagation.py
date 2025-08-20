import math
import random
'''
weighted sum is z
h is hidden layer
'''

LEARNING_RATE = 0.01
BIAS = 1

def sig(x):
    return 1 / (1 + math.exp(-x))

def sig_der(x):
    return x * (1 - x)

def cost(y_hat,y):
    return (y_hat-y)**2

def cost_der(y_hat,y):
    return 2*(y_hat-y)

def weighted_sum(bias, weights, inputs):
    weightSum = 0
    for i in range(len(inputs)):
        for j in range(len(inputs[i])):
            weightSum += inputs[i][j] * weights[i][j]
    return weightSum + bias
        
def out_err_grad(y_hat, y):
     return cost_der(y_hat, y) * sig_der(y_hat)







    