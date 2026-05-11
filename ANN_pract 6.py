# ANN Practical 6
# Forward Propagation and Backpropagation

import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid
def derivative(x):
    return x * (1 - x)

# XOR Input Data
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

# Target Output
Y = np.array([[0],
              [1],
              [1],
              [0]])

# Random weights
np.random.seed(1)

w1 = np.random.rand(2,4)
w2 = np.random.rand(4,1)

# Bias
b1 = np.random.rand(1,4)
b2 = np.random.rand(1,1)

# Training
epochs = 10000
lr = 0.1

for i in range(epochs):

    # Forward Propagation
    hidden = sigmoid(np.dot(X, w1) + b1)

    output = sigmoid(np.dot(hidden, w2) + b2)

    # Error
    error = Y - output

    # Backpropagation
    d_output = error * derivative(output)

    hidden_error = d_output.dot(w2.T)

    d_hidden = hidden_error * derivative(hidden)

    # Update weights and bias
    w2 += hidden.T.dot(d_output) * lr
    w1 += X.T.dot(d_hidden) * lr

    b2 += np.sum(d_output, axis=0) * lr
    b1 += np.sum(d_hidden, axis=0) * lr

# Testing
print("Final Output:")

for i in range(len(X)):

    hidden = sigmoid(np.dot(X[i], w1) + b1)

    output = sigmoid(np.dot(hidden, w2) + b2)

    print(X[i], "=", round(output[0][0]))



# Viva Questions & Answers

# Q1. What is forward propagation?
# Data moves from input layer to output layer.

# Q2. What is backpropagation?
# It updates weights using error.

# Q3. Why sigmoid is used?
# To get output between 0 and 1.

# Q4. What is epoch?
# One complete training cycle.

# Q5. What is learning rate?
# It controls learning speed.
