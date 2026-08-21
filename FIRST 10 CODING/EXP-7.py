import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Read CSV file
data = pd.read_csv(r"C:\Users\divya\Downloads\LogisticRegression.csv")

print("Training Data:\n")
print(data)

encoder = LabelEncoder()
encoded_data = data.copy()

for column in encoded_data.columns:
    encoded_data[column] = encoder.fit_transform(encoded_data[column])

X = encoded_data.iloc[:, :-1]
y = encoded_data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1
)

model = LogisticRegression()

model.fit(X_train, y_train)

prediction = model.predict(X_test)

print("\nPredicted Output:")
print(prediction)

print("\nActual Output:")
print(y_test.values)

print("\nAccuracy:", accuracy_score(y_test, prediction))
