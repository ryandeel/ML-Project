'''
things needed for the perceptron:
2 classes,
inputs for the xor case, points being (0,0) (0,1) (1,0) (1,1)
learning rate, bias, and weights

'''
import random
import numpy as np
import propagation as prop

# truth table
X = [(0,0), (0,1), (1,0), (1,1)]


class MultiLayerPerceptron:
    def __init__(self):
        # thanks python libraries for random
        # self.weights = np.random.rand(2) # commenting this, as it works for a singular layer if we want test it works but not for multiple

        # self.weights_hidden add these
        # self.weights_output add these

        # self.bias = 1
        self.bias_hidden = np.ones((1, 2)) # size of hidden layer
        self.bias_output = 1 # size of output layer
        print(f"Initialized weights: {self.weights}, bias: {self.bias}")

    def step_function(self, x):
        return 1 if x >= 0 else 0
    
    def predict(self, X):
        # calculate weighted sum multiplying weight vetor with input vector
        weighted_sum = np.dot(X, self.weights) + self.bias
        return self.step_function(weighted_sum)
    
    def train(self, X, y):
        maxEpochs = 10000
        epochs = 0

        for epochs in range(maxEpochs):
            totalErrors = 0
            for xi, target in zip(X, y):
                y_pred = self.predict(xi)
                error = target - y_pred
                if error != 0:
                    # stack overflow said i gotta turn it into np aray to multiply float by a sequence 
                    self.weights += error * prop.LEARNING_RATE * np.asarray(xi)
                    self.bias += error * prop.LEARNING_RATE
                    totalErrors += 1

            # chatgpt print statement for testing
            print(f"Epoch {epochs}, Weights: {self.weights}, Bias: {self.bias}, Errors: {totalErrors}")

            epochs += 1
            print("total errors: ", totalErrors)
            if totalErrors == 0:
                print("epoch #: ", epochs)
                break
    
    # def feed_forward(self):

    # def back_propagation(self):
    
    
# targets for above
y1 = [0, 1, 1, 1]
# targets for below
y2 = [1, 1, 1, 0]
# targets for xor problem
xor = [0, 1, 1, 0]