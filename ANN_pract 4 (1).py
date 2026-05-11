import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
Y = np.array([-1, -1, -1, 1])

w = np.zeros(X.shape[1])
b = 0

for _ in range(6):
    for i in range(X.shape[0]):
        y_pred = np.sign(np.dot(X[i], w) + b)
        
        if y_pred != Y[i]:
            w += 0.3 * Y[i] * X[i]
            b += 0.3 * Y[i]

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

Z = np.sign(np.dot(np.c_[xx.ravel(), yy.ravel()], w) + b)
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=Y)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Perceptron Decision Regions')
plt.show()




# ------------------------------------------------
# for viva

# # import numpy for mathematical operations
# import numpy as np

# # import matplotlib for plotting graph
# import matplotlib.pyplot as plt


# # input data
# X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])

# # target output
# # -1 and 1 are used for binary classification
# Y = np.array([-1, -1, -1, 1])


# # initialize weights with zeros
# w = np.zeros(X.shape[1])

# # initialize bias as 0
# b = 0


# # training perceptron model
# for _ in range(6):

#     # loop through all input samples
#     for i in range(X.shape[0]):

#         # calculate predicted output
#         y_pred = np.sign(np.dot(X[i], w) + b)

#         # check prediction error
#         if y_pred != Y[i]:

#             # update weights
#             w += 0.3 * Y[i] * X[i]

#             # update bias
#             b += 0.3 * Y[i]


# # define graph boundaries
# x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
# y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1


# # create mesh grid for decision region
# xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
#                      np.arange(y_min, y_max, 0.01))


# # predict values for all graph points
# Z = np.sign(np.dot(np.c_[xx.ravel(), yy.ravel()], w) + b)

# # reshape output into graph shape
# Z = Z.reshape(xx.shape)


# # plot decision regions
# plt.contourf(xx, yy, Z, alpha=0.8)

# # plot training data points
# plt.scatter(X[:, 0], X[:, 1], c=Y)

# # label x-axis
# plt.xlabel('X1')

# # label y-axis
# plt.ylabel('X2')

# # graph title
# plt.title('Perceptron Decision Regions')

# # show graph
# plt.show()


# # Viva Questions & Answers

# # Q1. What is perceptron learning law?
# # It updates weights based on prediction error.

# # Q2. Why np.sign() is used?
# # It converts output into -1 or 1 for classification.

# # Q3. What is decision region?
# # Area where model classifies data into different classes.

# # Q4. Why weights are updated?
# # To reduce prediction error and improve accuracy.

# # Q5. What is bias?
# # Bias helps shift the decision boundary.

# # Q6. Why contourf() is used?
# # It plots decision regions graphically.