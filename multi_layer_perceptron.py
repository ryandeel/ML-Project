'''
things needed for the perceptron:
2 classes,
inputs for the xor case, points being (0,0) (0,1) (1,0) (1,1)
learning rate, bias, and weights

'''
import random
import numpy as np
import propagation.py as prop

# truth table
X = [(0,0), (0,1), (1,0), (1,1)]


class MultiLayerPerceptron:
    def __init__(self):
        # thanks python libraries for random
        self.weights = np.random.rand(2)
        self.bias = random.uniform(0,1)

    def step_function(self, x):
        return 1 if x >= 0 else 0
    
    def predict(self, x):
        # calculate weighted sum multiplying weight vetor with input vector
        weighted_sum = np.dot(self.weights, x) + self.bias
        return self.step_function(weighted_sum)
    
    def train(self, X, y1):
        maxEpochs = 10000
        epochs = 0

        for epochs in range(maxEpochs):
            totalErrors = 0
            for xi, target in zip(X, y1):
                y_pred = self.predict(xi)
                error = target - y_pred
                if error != 0:
                    self.weights += error * prop.LEARNING_RATE * xi
                    self.bias += error * prop.LEARNING_RATE
                    totalErrors += 1

            epochs += 1
            print("total errors: ", totalErrors)
            if totalErrors == 0:
                print("epoch #: ", epochs)
                break
    
# targets for above
y1 = [0, 1, 1, 1]
# targets for below
y2 = [1, 1, 1, 0]