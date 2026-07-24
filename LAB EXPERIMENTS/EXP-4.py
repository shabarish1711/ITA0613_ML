# BACKPROPAGATION ALGORITHM

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Read the CSV file
data = pd.read_csv(r"C:\Users\divya\Downloads\Backpropagation.csv")

# Display the dataset
print("Training Data:\n")
print(data)

# Convert categorical data into numerical data
encoder = LabelEncoder()
encoded_data = data.copy()

for column in encoded_data.columns:
    encoded_data[column] = encoder.fit_transform(encoded_data[column])

# Separate input and output
X = encoded_data.iloc[:, :-1]
y = encoded_data.iloc[:, -1]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1
)

# Create ANN Model
model = MLPClassifier(hidden_layer_sizes=(5,), max_iter=1000, random_state=1)

# Train the model
model.fit(X_train, y_train)

# Predict the output
prediction = model.predict(X_test)

# Display results
print("\nPredicted Output:")
print(prediction)

print("\nActual Output:")
print(y_test.values)

# Display accuracy
accuracy = accuracy_score(y_test, prediction)
print("\nAccuracy:", accuracy)
