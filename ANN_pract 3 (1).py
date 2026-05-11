# Simple Perceptron for Even-Odd Recognition (ASCII digits)


# step activation function
def step(x):

    # return 1 if value >= 0 otherwise 0
    return 1 if x >= 0 else 0


# create empty training data list
training_data = []


# take number of training samples
n = int(input("Enter number of training samples: "))


print("Enter ASCII value and target (0=Even, 1=Odd)")


# input training data
for i in range(n):

    # take ASCII value
    ascii_val = int(input("Enter ASCII value (48-57): "))

    # take target output
    target = int(input("Enter target (0 for Even, 1 for Odd): "))

    # feature extraction using modulo
    x = ascii_val % 2

    # store data
    training_data.append((x, target))


# initialize weight and bias
w = 0.0
b = 0.0

# learning rate
learning_rate = 0.1


# training perceptron model
for epoch in range(10):

    for x, target in training_data:

        # calculate output
        y = step(w * x + b)

        # calculate error
        error = target - y

        # update weight
        w += learning_rate * error * x

        # update bias
        b += learning_rate * error


# testing phase
print("\nASCII  Digit  Result")


for ascii_val in range(48, 58):

    # extract feature
    x = ascii_val % 2

    # predict output
    output = step(w * x + b)

    # convert ASCII to digit
    digit = chr(ascii_val)

    # display result
    result = "Odd" if output == 1 else "Even"

    print(ascii_val, "   ", digit, "   ", result)


# Viva Questions & Answers

# Q1. What is perceptron?
# Perceptron is a single-layer neural network.

# Q2. Why ASCII values are used?
# To represent digits numerically.

# Q3. Why modulo (%) is used?
# To identify even and odd numbers.

# Q4. What is learning rate?
# It controls speed of learning.

# Q5. What is step activation function?
# It converts output into binary form 0 or 1.

# Q6. What is bias?
# Bias helps shift decision boundary.
