'''
things needed for the perceptron:
2 classes,
inputs for the xor case, points being (0,0) (0,1) (1,0) (1,1)
learning rate, bias, and weights

HI BLAISE and future me ok so im going to bed but like im not sure making two perceptron classes was the move ngl
^ i think i should be using one general class and make two? i think the data is where the problem goes but 
since u know more than me abt this blaise i think u will know what im doing wrong

i got the above one to work but not the below one

'''
import math
import random

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

LEARNING_RATE = 0.01

# truth table
X = [(0,0), (0,1), (1,0), (1,1)]


class Perceptron:
    def __init__(self):
        # thanks python libraries for random
        self.x_weight = random.uniform(0,1)
        self.y_weight = random.uniform(0,1)
        self.bias = random.uniform(0,1)

    def step_function(self, x):
        return 1 if x >= 0 else 0
    
    def predict(self, x_input, y_input):
        # calculate weighted sum multiplying weight vetor with input vector
        weighted_sum = self.x_weight * x_input + self.y_weight * y_input + self.bias
        return self.step_function(weighted_sum)
    
    def train(self, X, y1):
        maxEpochs = 10000
        epochs = 0

        while epochs < maxEpochs:
            totalErrors = 0
            for (x_input, y_input), target in zip(X, y1):
                y_pred = self.predict(x_input, y_input)
                error = target - y_pred
                if error != 0:
                    self.x_weight += error * LEARNING_RATE * x_input
                    self.y_weight += error * LEARNING_RATE * y_input
                    self.bias += error * LEARNING_RATE
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

p_test_above = Perceptron()
p_test_above.train(X, y1)

# chatgpt created print line modified to test above perceptron
for x_input, y_input in X:
    print(f"Input: ({x_input}, {y_input}), Prediction: {p_test_above.predict(x_input, y_input)}")

p_test_below = Perceptron()
p_test_below.train(X, y2)

# chatgpt created print line modified to test below perceptron
for x_input, y_input in X:
    print(f"Input: ({x_input}, {y_input}), Prediction: {p_test_below.predict(x_input, y_input)}")