import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Read CSV file
data = pd.read_csv(r"C:\Users\divya\Downloads\PolynomialRegression.csv")

print("Training Data:\n")
print(data)

X = data[['Experience']]
y = data['Salary']

linear = LinearRegression()
linear.fit(X, y)

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

poly_model = LinearRegression()
poly_model.fit(X_poly, y)

linear_prediction = linear.predict([[6]])
poly_prediction = poly_model.predict(poly.transform([[6]]))

print("\nLinear Regression Prediction:")
print(linear_prediction)

print("\nPolynomial Regression Prediction:")
print(poly_prediction)
