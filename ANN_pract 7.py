# ANN Practical 7
# XOR using Backpropagation

import numpy as np

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative
def derivative(x):
    return x * (1 - x)

# XOR data
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

Y = np.array([[0],
              [1],
              [1],
              [0]])

# Random weights
np.random.seed(1)

w1 = np.random.rand(2,2)
w2 = np.random.rand(2,1)

# Training
for i in range(5000):

    # Forward propagation
    hidden = sigmoid(np.dot(X, w1))

    output = sigmoid(np.dot(hidden, w2))

    # Error
    error = Y - output

    # Backpropagation
    d_output = error * derivative(output)

    d_hidden = d_output.dot(w2.T) * derivative(hidden)

    # Update weights
    w2 += hidden.T.dot(d_output) * 0.1
    w1 += X.T.dot(d_hidden) * 0.1

# Output
print("XOR Output:")

for i in range(len(X)):

    hidden = sigmoid(np.dot(X[i], w1))

    output = sigmoid(np.dot(hidden, w2))

    print(X[i], "=", round(output[0][0]))



# Viva Questions & Answers

# Q1. What is XOR?
# Output is 1 when inputs are different.

# Q2. What is backpropagation?
# It updates weights using error.

# Q3. Why sigmoid is used?
# To get output between 0 and 1.

# Q4. Why hidden layer is used?
# To solve non-linear problems.
