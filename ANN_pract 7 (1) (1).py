# ANN Practical 7
# Backpropagation Network for XOR Function

import numpy as np

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid
def derivative(x):
    return x * (1 - x)

# XOR Input
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

# XOR Output
Y = np.array([[0],
              [1],
              [1],
              [0]])

# Random weights and bias
np.random.seed(1)

W1 = np.random.rand(2,4)
W2 = np.random.rand(4,1)

b1 = np.random.rand(1,4)
b2 = np.random.rand(1,1)

# Training
epochs = 10000
lr = 0.1

for i in range(epochs):

    # Forward propagation
    hidden = sigmoid(np.dot(X, W1) + b1)

    output = sigmoid(np.dot(hidden, W2) + b2)

    # Error
    error = Y - output

    # Backpropagation
    d_output = error * derivative(output)

    hidden_error = d_output.dot(W2.T)

    d_hidden = hidden_error * derivative(hidden)

    # Update weights and bias
    W2 += hidden.T.dot(d_output) * lr
    W1 += X.T.dot(d_hidden) * lr

    b2 += np.sum(d_output, axis=0) * lr
    b1 += np.sum(d_hidden, axis=0) * lr

# Final Output
print("XOR Output:")

for i in range(len(X)):

    hidden = sigmoid(np.dot(X[i], W1) + b1)

    output = sigmoid(np.dot(hidden, W2) + b2)

    print(X[i], "=", round(output[0][0]))



# Viva Questions & Answers

# Q1. What is XOR function?
# XOR gives 1 when inputs are different.

# Q2. Why perceptron cannot solve XOR?
# Because XOR is non-linear.

# Q3. Why hidden layer is used?
# To solve complex problems.

# Q4. What is backpropagation?
# It updates weights using error.

# Q5. Why sigmoid is used?
# To get output between 0 and 1.
