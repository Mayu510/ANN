# ANN Practical 4
# Perceptron Learning Law with Decision Region

import numpy as np
import matplotlib.pyplot as plt

# Input data
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

# Target output
Y = np.array([-1, -1, -1, 1])

# Initialize weights and bias
w = np.zeros(2)
b = 0

# Train perceptron
for i in range(6):

    for j in range(len(X)):

        y_pred = np.sign(np.dot(X[j], w) + b)

        # Update weights if prediction is wrong
        if y_pred != Y[j]:

            w = w + 0.3 * Y[j] * X[j]
            b = b + 0.3 * Y[j]

# Create decision region
x1, x2 = np.meshgrid(np.arange(-1, 2, 0.01),
                     np.arange(-1, 2, 0.01))

z = np.sign(w[0] * x1 + w[1] * x2 + b)

# Plot graph
plt.contourf(x1, x2, z, alpha=0.5)

plt.scatter(X[:,0], X[:,1], c=Y)

plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Perceptron Decision Region")

plt.show()



# Viva Questions & Answers

# Q1. What is perceptron learning law?
# It updates weights using error.

# Q2. What is decision region?
# Region used to classify data into classes.

# Q3. Why np.sign() is used?
# To give output as -1 or 1.

# Q4. What is bias?
# Bias shifts decision boundary.

# Q5. Why weights are updated?
# To improve prediction accuracy.
