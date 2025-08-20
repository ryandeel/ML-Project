import math
import random
'''
weighted sum is z
h is hidden layer

That VEINY DIH <3 got me MOANIN'
'''

LEARNING_RATE = 0.01

def sig(weight_sum):
    return 1 / (1 + math.exp(-weight_sum))

def sig_der(y_hat):
    return y_hat * (1 - y_hat)

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
        
def out_err_grad(cost_derivative, sigmoid_derivative):
     return cost_derivative * sigmoid_derivative

def hid_err_grad(output_error_gradient, output_weight, sigmoid_derivative):
    return output_error_gradient * output_weight * sigmoid_derivative

def out_weight_grad(output_error_gradient, hidden_layer):
    #hidden layer likely just going to be passed as a sigmoid of a weighted sum
    return output_error_gradient * hidden_layer

def hid_weight_grad(corresponding_input, hidden_error_gradient):
    return hidden_error_gradient * corresponding_input
    
def update_bias(old_bias, gradient):
    return old_bias - LEARNING_RATE*gradient





    