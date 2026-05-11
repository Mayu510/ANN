import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 100)
plt.plot(x, 1 / (1 + np.exp(-x)), label='Sigmoid')
plt.plot(x, np.tanh(x), label='tanh')
plt.plot(x, np.maximum(0, x), label='ReLU')
plt.plot(x, x, label='Identity')
plt.plot(x, np.exp(x) / np.sum(np.exp(x)), label='Softmax')

plt.xlabel('Input')
plt.ylabel('Activation')
plt.title('Activation Functions')
plt.legend()
plt.grid(True)
plt.show()

#-------------------------------------------------------------------

# for viva
# # import numpy for mathematical operations
# import numpy as np

# # import matplotlib for plotting graphs
# import matplotlib.pyplot as plt

# # create 100 input values between -5 and 5
# x = np.linspace(-5, 5, 100)

# # plot Sigmoid activation function (output range 0 to 1)
# plt.plot(x, 1 / (1 + np.exp(-x)), label='Sigmoid')

# # plot Tanh activation function (output range -1 to 1)
# plt.plot(x, np.tanh(x), label='tanh')

# # plot ReLU activation function (negative values become 0)
# plt.plot(x, np.maximum(0, x), label='ReLU')

# # plot Identity function (output = input)
# plt.plot(x, x, label='Identity')

# # plot Softmax activation function (converts values into probabilities)
# plt.plot(x, np.exp(x) / np.sum(np.exp(x)), label='Softmax')

# # label x-axis
# plt.xlabel('Input')

# # label y-axis
# plt.ylabel('Activation')

# # give title to graph
# plt.title('Activation Functions')

# # show function names
# plt.legend()

# # display grid
# plt.grid(True)

# # show final graph
# plt.show()


# # Viva Questions & Answers

# # Q1. What is activation function?
# # Activation function adds non-linearity to neural networks.

# # Q2. Why ReLU is mostly used?
# # Because it is fast and computationally efficient.

# # Q3. Why softmax is used?
# # Softmax is used for multiclass classification.

# # Q4. Difference between Sigmoid and Tanh?
# # Sigmoid range: 0 to 1
# # Tanh range: -1 to 1