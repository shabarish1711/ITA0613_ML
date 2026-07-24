import pandas as pd
from sklearn.mixture import GaussianMixture

# Read CSV file
data = pd.read_csv(r"C:\Users\divya\Downloads\ExpectationMaximization.csv")

print("Training Data:\n")
print(data)

X = data[['Height', 'Weight']]

model = GaussianMixture(n_components=2, random_state=1)

model.fit(X)

prediction = model.predict(X)

print("\nCluster Labels:")
print(prediction)
