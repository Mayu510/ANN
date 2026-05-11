import numpy as np
def mp_neuron(inputs, weights, threshold):
    weighted_sum = np.dot(inputs, weights)
    output = 1 if weighted_sum >= threshold else 0
    return output
def and_not(x1, x2):
    weights = [1, -1] 
    threshold = 1   
    inputs = np.array([x1, x2])
    output = mp_neuron(inputs, weights, threshold)
    return output
print(and_not(0, 0)) 
print(and_not(1, 0))  
print(and_not(0, 1))  
print(and_not(1, 1))  


# ---------------------------------------------------------------

# for viva
# # import numpy library for mathematical operations
# import numpy as np

# # function for McCulloch-Pitts neuron
# def mp_neuron(inputs, weights, threshold):

#     # calculate weighted sum using dot product
#     weighted_sum = np.dot(inputs, weights)

#     # apply threshold activation function
#     output = 1 if weighted_sum >= threshold else 0

#     # return final output
#     return output


# # function for ANDNOT logic gate
# def and_not(x1, x2):

#     # assign weights
#     weights = [1, -1]

#     # set threshold value
#     threshold = 1

#     # convert inputs into array
#     inputs = np.array([x1, x2])

#     # call mp neuron function
#     output = mp_neuron(inputs, weights, threshold)

#     # return output
#     return output


# # test cases for ANDNOT gate
# print(and_not(0, 0))   # output = 0
# print(and_not(1, 0))   # output = 1
# print(and_not(0, 1))   # output = 0
# print(and_not(1, 1))   # output = 0


# # Viva Questions & Answers

# # Q1. What is McCulloch-Pitts neuron?
# # It is the first artificial neuron model with binary output.

# # Q2. What is weighted sum?
# # Sum of input values multiplied by their weights.

# # Q3. Why threshold is used?
# # To decide whether neuron activates or not.

# # Q4. What is ANDNOT gate?
# # It gives output 1 only when x1=1 and x2=0.

# # Q5. Why np.dot() is used?
# # It calculates multiplication and addition of inputs and weights.