import pandas as pd
from sklearn.linear_model import LinearRegression

# Read CSV file
data = pd.read_csv(r"C:\Users\divya\Downloads\LinearRegression.csv")

print("Training Data:\n")
print(data)

X = data[['Hours']]
y = data['Marks']

model = LinearRegression()

model.fit(X, y)

prediction = model.predict([[7]])

print("\nPredicted Marks for 7 Hours:")
print(prediction)
