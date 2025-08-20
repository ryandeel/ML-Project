import math
import random

LEARNING_RATE = 0.01
BIAS = 1

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def cost(y_hat,y):
    return (y_hat-y)**2

def cost_der(i,y_hat,y):
    return 2*i*(y_hat-y)

def weighted_sum(bias, weights, inputs):
    sum = 0
    for input in range(inputs):
        sum += input * weights[input]
    return sum + bias
        







    