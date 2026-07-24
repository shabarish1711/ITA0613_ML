# FIND-S ALGORITHM

import pandas as pd

# Read the CSV file
data = pd.read_csv(r"C:\Users\divya\Downloads\FindS.csv")

# Display the dataset
print("Training Data:\n")
print(data)

# Separate attributes and target
concepts = data.iloc[:, :-1].values
target = data.iloc[:, -1].values

# Initialize the hypothesis
hypothesis = ['0'] * len(concepts[0])

# Apply Find-S Algorithm
for i in range(len(concepts)):
    if target[i] == "Yes":
        for j in range(len(hypothesis)):
            if hypothesis[j] == '0':
                hypothesis[j] = concepts[i][j]
            elif hypothesis[j] != concepts[i][j]:
                hypothesis[j] = '?'

# Display the final hypothesis
print("\nMost Specific Hypothesis:")
print(hypothesis)
