# ANN Practical 5
# Bidirectional Associative Memory (BAM)

import numpy as np

# Input patterns
X = np.array([[1, 1, 1, -1],
              [-1, -1, 1, 1]])

# Output patterns
Y = np.array([[1, -1],
              [-1, 1]])

# Create weight matrix
W = np.dot(Y.T, X)

# Sign activation function
def sign(x):
    return np.where(x >= 0, 1, -1)

# BAM recall function
def bam(x):

    for i in range(10):

        # X to Y
        y = sign(np.dot(W, x))

        # Y to X
        x_new = sign(np.dot(W.T, y))

        # Stop if output becomes stable
        if np.array_equal(x, x_new):
            break

        x = x_new

    return x, y


# User Input
# Example Input:
# Enter 4 values: 1 -1 -1 -1

x_test = np.array(list(map(int,
            input("Enter 4 values separated by space: ").split())))

# Run BAM Network
x_final, y_final = bam(x_test)

# Output
print("Final X:", x_final)
print("Final Y:", y_final)



# Viva Questions & Answers

# Q1. What is BAM?
# BAM stands for Bidirectional Associative Memory.

# Q2. Why it is called bidirectional?
# Because data moves in both directions.

# Q3. What is weight matrix?
# It stores relation between patterns.

# Q4. Why sign function is used?
# To convert output into +1 and -1.

# Q5. What is convergence?
# When output becomes stable.
