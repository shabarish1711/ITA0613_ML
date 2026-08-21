# ID3 DECISION TREE ALGORITHM

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# Read the CSV file
data = pd.read_csv(r"C:\Users\divya\Downloads\ID3.csv")

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

# Create Decision Tree using Entropy (ID3)
model = DecisionTreeClassifier(criterion="entropy")

# Train the model
model.fit(X, y)

# Test with the first sample
sample = X.iloc[[0]]

prediction = model.predict(sample)

print("\nPrediction for First Sample:")
print(prediction)
