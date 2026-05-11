import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Normalize the data (scale pixel values to 0-1)
X_train, X_test = X_train / 255.0, X_test / 255.0

# Build a simple CNN model
model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')  # 10 classes (digits 0-9)
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_accuracy)

# Predict a random sample
index = np.random.randint(0, len(X_test))
plt.imshow(X_test[index], cmap='gray')
plt.title(f"True Label: {y_test[index]}")
plt.show()

prediction = np.argmax(model.predict(X_test[index].reshape(1, 28, 28, 1)))
print("Predicted Label:", prediction)


# -----------------------------------------------
# for viva
# # import tensorflow and keras
# import tensorflow as tf
# from tensorflow import keras

# # import libraries
# import matplotlib.pyplot as plt
# import numpy as np


# # load MNIST dataset
# (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()


# # normalize data
# X_train = X_train / 255.0
# X_test = X_test / 255.0


# # build CNN model
# model = keras.Sequential([

#     keras.layers.Conv2D(
#         32,
#         (3,3),
#         activation='relu',
#         input_shape=(28,28,1)
#     ),

#     keras.layers.MaxPooling2D(2,2),

#     keras.layers.Conv2D(
#         64,
#         (3,3),
#         activation='relu'
#     ),

#     keras.layers.MaxPooling2D(2,2),

#     keras.layers.Flatten(),

#     keras.layers.Dense(128, activation='relu'),

#     keras.layers.Dense(10, activation='softmax')
# ])


# # compile model
# model.compile(
#     optimizer='adam',
#     loss='sparse_categorical_crossentropy',
#     metrics=['accuracy']
# )


# # train model
# model.fit(X_train, y_train, epochs=5,
#           validation_data=(X_test, y_test))


# # evaluate model
# test_loss, test_accuracy = model.evaluate(X_test, y_test)

# print("Test Accuracy:", test_accuracy)


# # select random image
# index = np.random.randint(0, len(X_test))


# # show image
# plt.imshow(X_test[index], cmap='gray')

# plt.title(f"True Label: {y_test[index]}")

# plt.show()


# # predict digit
# prediction = np.argmax(
#     model.predict(X_test[index].reshape(1, 28, 28, 1))
# )

# print("Predicted Label:", prediction)


# # Viva Questions & Answers

# # Q1. What is CNN?
# # CNN is Convolutional Neural Network used for image processing.

# # Q2. What is MNIST?
# # Dataset of handwritten digits from 0 to 9.

# # Q3. Why normalization is done?
# # To scale pixel values between 0 and 1.

# # Q4. Why softmax is used?
# # For multiclass classification.

# # Q5. What is Conv2D layer?
# # It extracts image features using filters.