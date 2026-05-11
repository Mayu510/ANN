import numpy as np

# Step function (activation)
step_function = lambda x: 1 if x >= 0 else 0

# ASCII training data for digits '0' to '9'
# ASCII of '0' = 48 (binary: 0110000), up to '9' = 57 (binary: 0111001)
training_data = []

for i in range(10):
    char = chr(48 + i)  # Get ASCII character for digits 0 to 9
    ascii_value = ord(char)
    binary_input = [int(bit) for bit in '{0:07b}'.format(ascii_value)]  # 7-bit binary
    label = 1 if i % 2 == 0 else 0  # 1 = even, 0 = odd
    training_data.append({'input': binary_input, 'label': label})

# Initialize weights (same size as input vector)
weights = np.zeros(7)

# Training loop (single pass)
for data in training_data:
    input_vector = np.array(data['input'])
    label = data['label']
    output = step_function(np.dot(input_vector, weights))
    error = label - output
    weights += input_vector * error  # Perceptron update rule

# User input
user_char = input("Enter a digit between 0 and 9: ")

# Validate input
if user_char not in "0123456789":
    print("Invalid input. Please enter a digit between 0 and 9.")
else:
    ascii_input = ord(user_char)
    binary_input = np.array([int(bit) for bit in '{0:07b}'.format(ascii_input)])
    prediction = step_function(np.dot(binary_input, weights))
    result = "even" if prediction == 1 else "odd"
    print(f"{user_char} (ASCII: {ascii_input}) is {result}.")





# -------------------------------
# for viva

# # import numpy library for mathematical operations
# import numpy as np

# # define step activation function
# # if value >= 0 output is 1 otherwise 0
# step_function = lambda x: 1 if x >= 0 else 0


# # create empty list for training data
# training_data = []

# # loop for digits 0 to 9
# for i in range(10):

#     # convert number to ASCII character
#     char = chr(48 + i)

#     # get ASCII value
#     ascii_value = ord(char)

#     # convert ASCII value into 7-bit binary
#     binary_input = [int(bit) for bit in '{0:07b}'.format(ascii_value)]

#     # assign label
#     # 1 for even number
#     # 0 for odd number
#     label = 1 if i % 2 == 0 else 0

#     # store input and label in training data
#     training_data.append({'input': binary_input, 'label': label})


# # initialize weights with zeros
# weights = np.zeros(7)


# # training perceptron model
# for data in training_data:

#     # convert input into array
#     input_vector = np.array(data['input'])

#     # get actual label
#     label = data['label']

#     # calculate output using weighted sum
#     output = step_function(np.dot(input_vector, weights))

#     # calculate error
#     error = label - output

#     # update weights using perceptron learning rule
#     weights += input_vector * error


# # take user input
# user_char = input("Enter a digit between 0 and 9: ")


# # check valid input
# if user_char not in "0123456789":

#     # print invalid message
#     print("Invalid input. Please enter a digit between 0 and 9.")

# else:

#     # get ASCII value of user input
#     ascii_input = ord(user_char)

#     # convert ASCII into binary
#     binary_input = np.array([int(bit) for bit in '{0:07b}'.format(ascii_input)])

#     # predict even or odd
#     prediction = step_function(np.dot(binary_input, weights))

#     # display result
#     result = "even" if prediction == 1 else "odd"

#     print(f"{user_char} (ASCII: {ascii_input}) is {result}.")


# # Viva Questions & Answers

# # Q1. What is Perceptron?
# # Perceptron is a single layer neural network used for binary classification.

# # Q2. Why step function is used?
# # It converts output into binary values 0 or 1.

# # Q3. What is ASCII?
# # ASCII is a character encoding standard for numbers and symbols.

# # Q4. Why binary conversion is done?
# # Neural networks work with numerical binary input data.

# # Q5. What is perceptron learning rule?
# # It updates weights using error value to improve prediction.

# # Q6. What does np.dot() do?
# # It calculates weighted sum of inputs and weights.