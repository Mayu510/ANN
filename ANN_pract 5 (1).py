import numpy as np

# Training data
X = np.array([[1, 1, 1, -1],
              [-1, -1, 1, 1]])
Y = np.array([[1, -1],
              [-1, 1]])

# Weight matrix using outer product sum
W = np.dot(Y.T, X)

# Sign function that handles zero
def custom_sign(x):
    return np.where(x >= 0, 1, -1)

# BAM recall function (iterates until convergence)
def bam_bidirectional(x_init, max_iters=10):
    x = x_init.copy()
    for _ in range(max_iters):
        y = custom_sign(np.dot(W, x))          # X to Y
        x_new = custom_sign(np.dot(W.T, y))    # Y to X
        if np.array_equal(x, x_new):           # Check for convergence
            break
        x = x_new
    return x, y

# Test input
x_test = np.array([1, -1, -1, -1])

# Run BAM
x_final, y_final = bam_bidirectional(x_test)

# Print results
print("Initial input x:", x_test)
print("Final recalled x:", x_final)
print("Recalled y:", y_final)




# ----------------------------------------------------
# for viva
# # import numpy library
# import numpy as np


# # training input patterns
# X = np.array([[1, 1, 1, -1],
#               [-1, -1, 1, 1]])

# # target output patterns
# Y = np.array([[1, -1],
#               [-1, 1]])


# # create weight matrix using dot product
# W = np.dot(Y.T, X)


# # custom sign activation function
# # returns 1 for positive values and -1 for negative values
# def custom_sign(x):
#     return np.where(x >= 0, 1, -1)


# # BAM recall function
# def bam_bidirectional(x_init, max_iters=10):

#     # copy input pattern
#     x = x_init.copy()

#     # repeat until convergence
#     for _ in range(max_iters):

#         # propagate from X layer to Y layer
#         y = custom_sign(np.dot(W, x))

#         # propagate from Y layer back to X layer
#         x_new = custom_sign(np.dot(W.T, y))

#         # stop if output becomes stable
#         if np.array_equal(x, x_new):
#             break

#         # update x
#         x = x_new

#     # return final patterns
#     return x, y


# # test input pattern
# x_test = np.array([1, -1, -1, -1])


# # run BAM network
# x_final, y_final = bam_bidirectional(x_test)


# # print outputs
# print("Initial input x:", x_test)
# print("Final recalled x:", x_final)
# print("Recalled y:", y_final)


# # Viva Questions & Answers

# # Q1. What is BAM?
# # BAM stands for Bidirectional Associative Memory.

# # Q2. Why BAM is called bidirectional?
# # Because data flows from X to Y and Y to X.

# # Q3. What is weight matrix?
# # It stores association between input and output patterns.

# # Q4. Why sign function is used?
# # It converts output into bipolar values (+1 and -1).

# # Q5. What is convergence?
# # When output becomes stable and stops changing.

# # Q6. What type of network is BAM?
# # BAM is a recurrent neural network.