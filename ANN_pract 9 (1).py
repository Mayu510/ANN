import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

# Load the breast cancer dataset
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Scale the input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(X_train.shape[1],))
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Accuracy:", test_accuracy)


# ---------------------------------------
# for viva
# # import tensorflow and sklearn libraries
# import tensorflow as tf

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.datasets import load_breast_cancer


# # load dataset
# data = load_breast_cancer()


# # split training and testing data
# X_train, X_test, y_train, y_test = train_test_split(
#     data.data,
#     data.target,
#     test_size=0.2,
#     random_state=42
# )


# # normalize data
# scaler = StandardScaler()

# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)


# # build neural network model
# model = tf.keras.models.Sequential([

#     tf.keras.layers.Dense(
#         1,
#         activation='sigmoid',
#         input_shape=(X_train.shape[1],)
#     )
# ])


# # compile model
# model.compile(
#     optimizer='adam',
#     loss='binary_crossentropy',
#     metrics=['accuracy']
# )


# # train model
# model.fit(X_train, y_train, epochs=5)


# # evaluate model
# test_loss, test_accuracy = model.evaluate(X_test, y_test)

# print("Accuracy:", test_accuracy)


# # Viva Questions & Answers

# # Q1. What is TensorFlow?
# # TensorFlow is a deep learning framework by Google.

# # Q2. Why sigmoid is used?
# # For binary classification output.

# # Q3. What is optimizer?
# # It updates weights to reduce error.

# # Q4. Why scaling is done?
# # To improve model performance.

# # Q5. What is epoch?
# # One complete training cycle.