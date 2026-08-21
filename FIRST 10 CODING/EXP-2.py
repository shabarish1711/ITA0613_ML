# CANDIDATE ELIMINATION ALGORITHM

import pandas as pd

# Read the CSV file
data = pd.read_csv(r"C:\Users\divya\Downloads\CandidateElimination.csv")

# Display the dataset
print("Training Data:\n")
print(data)

# Separate attributes and target
concepts = data.iloc[:, :-1].values
target = data.iloc[:, -1].values

# Initialize Specific Hypothesis
specific = concepts[0].copy()

# Initialize General Hypothesis
general = [['?' for i in range(len(specific))] for j in range(len(specific))]

# Apply Candidate Elimination Algorithm
for i in range(len(concepts)):

    if target[i] == "Yes":
        for j in range(len(specific)):
            if concepts[i][j] != specific[j]:
                specific[j] = '?'
                general[j][j] = '?'

    elif target[i] == "No":
        for j in range(len(specific)):
            if concepts[i][j] != specific[j]:
                general[j][j] = specific[j]
            else:
                general[j][j] = '?'

# Remove empty hypotheses
general = [g for g in general if g != ['?'] * len(specific)]

# Display the result
print("\nSpecific Hypothesis:")
print(specific)

print("\nGeneral Hypothesis:")
for g in general:
    print(g)
