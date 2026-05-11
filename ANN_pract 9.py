# ANN Practical 9
# Neural Network using TensorFlow

import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
data = load_breast_cancer()

X = data.data
Y = data.target

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create model
model = tf.keras.Sequential([

    tf.keras.layers.Dense(1,
                          activation='sigmoid',
                          input_shape=(X_train.shape[1],))
])

# Compile model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(X_train, Y_train, epochs=5)

# Test model
loss, accuracy = model.evaluate(X_test, Y_test)

print("Accuracy:", accuracy)



# Viva Questions & Answers

# Q1. What is TensorFlow?
# TensorFlow is a deep learning library.

# Q2. Why sigmoid is used?
# For binary classification.

# Q3. What is optimizer?
# It updates weights to reduce error.

# Q4. Why scaling is done?
# To improve performance.

# Q5. What is epoch?
# One complete training cycle.
