# ANN Practical 8
# Hopfield Network to Store 4 Vectors

import numpy as np

# Stored vectors
vectors = np.array([[1, 1, -1, -1],
                    [-1, -1, 1, 1],
                    [1, -1, 1, -1],
                    [-1, 1, -1, 1]])

# Initialize weight matrix
W = np.zeros((4,4))

# Calculate weights
for i in range(4):

    for j in range(4):

        # No self connection
        if i != j:

            W[i][j] = np.sum(vectors[i] * vectors[j])

# Activation function
def sign(x):
    return np.where(x >= 0, 1, -1)

# Hopfield Network
def hopfield(x):

    y = np.dot(W, x)

    return sign(y)

# User Input
# Example Input:
# Enter 4 values: 1 -1 1 -1

x = np.array(list(map(int,
        input("Enter 4 values separated by space: ").split())))

# Get output
output = hopfield(x)

# Print result
print("Output Vector:")
print(output)



# Viva Questions & Answers

# Q1. What is Hopfield Network?
# It is a recurrent neural network.

# Q2. Why self connection is avoided?
# Neuron should not connect to itself.

# Q3. What type of output is generated?
# Bipolar output (+1 and -1).

# Q4. What is associative memory?
# It recalls stored patterns.
