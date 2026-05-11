# ANN Practical 3
# Perceptron Neural Network for Even and Odd Recognition

import numpy as np

# Step activation function
def step(x):
    return 1 if x >= 0 else 0

training_data = []

# Create training data for digits 0 to 9
for i in range(10):

    ascii_value = ord(str(i))   # Get ASCII value
    binary = [int(bit) for bit in format(ascii_value, '07b')]  # Convert to binary

    label = 1 if i % 2 == 0 else 0   # 1 = Even, 0 = Odd

    training_data.append([binary, label])

# Initialize weights
weights = np.zeros(7)

# Train perceptron
for data in training_data:

    x = np.array(data[0])
    y = data[1]

    output = step(np.dot(x, weights))

    error = y - output

    weights = weights + x * error

# User Input
# Example Input:
# Enter number: 6

num = input("Enter a digit (0-9): ")

# Check valid input
if num.isdigit() and int(num) <= 9:

    ascii_value = ord(num)

    binary = np.array([int(bit) for bit in format(ascii_value, '07b')])

    prediction = step(np.dot(binary, weights))

    if prediction == 1:
        print(num, "is Even")
    else:
        print(num, "is Odd")

else:
    print("Invalid Input")



# Viva Questions & Answers

# Q1. What is Perceptron?
# Perceptron is a single layer neural network.

# Q2. Why step function is used?
# To give binary output 0 or 1.

# Q3. What is ASCII?
# ASCII is a character encoding system.

# Q4. Why binary conversion is used?
# Neural networks work on numerical data.

# Q5. What is perceptron learning rule?
# It updates weights using error value.
