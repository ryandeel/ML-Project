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
    weight_sum = 0
    for i in range(len(inputs)):
        for j in range(len(inputs[i])):
            weight_sum += inputs[i][j] * weights[i][j]
    return weight_sum + bias
        
def out_err_grad(y_hat, y):
     return cost_der(y_hat, y) * sig_der(y_hat)

def hid_err_grad(y_hat, y, output_weight, weight_sum):
    return out_err_grad(y_hat, y) * output_weight * sig_der(sig(weight_sum))

def out_weight_grad(y_hat, y, weight_sum):
    return out_err_grad(y_hat, y) * sig(weight_sum)

def hid_weight_grad(input, y_hat, y, output_weight, weight_sum):
    return hid_err_grad(y_hat, y, output_weight, weight_sum) * input
    






    