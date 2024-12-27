import numpy as np

# Step activation function
def step_function(x):
    return np.where(x >= 0, 1, 0)

# Training data for OR gate
# Inputs: [[0, 0], [0, 1], [1, 0], [1, 1]]
# Outputs: [0, 1, 1, 1]
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([0, 1, 1, 1])

# Initialize weights and bias
weights = np.random.uniform(-1, 1, size=2)
bias = np.random.uniform(-1, 1)

# Training parameters
learning_rate = 0.1
num_epochs = 20

# Training loop
for epoch in range(num_epochs):
    total_error = 0
    for x, target in zip(inputs, targets):
        # Compute weighted sum
        linear_output = np.dot(x, weights) + bias
        # Apply activation function
        prediction = step_function(linear_output)
        # Compute error
        error = target - prediction
        total_error += abs(error)
        # Update weights and bias manually
        weights += learning_rate * error * x
        bias += learning_rate * error
    print(f"Epoch {epoch + 1}/{num_epochs}, Total Error: {total_error}")

# Testing the model
print("\nTesting the OR Gate Neural Network:")
for x in inputs:
    linear_output = np.dot(x, weights) + bias
    prediction = step_function(linear_output)
    print(f"Input: {x}, Prediction: {prediction}")
