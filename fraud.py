import seaborn
import pandas as pd
import numpy as np
import codecademylib3
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import codecademylib3

# Load the data
df = pd.read_csv('transactions_modified.csv')
# print(df.head())
# print(df.info())

# How many fraudulent transactions?
frauds = df['isFraud'].sum()
# print(frauds)

# Summary statistics on amount column
# print(df['amount'].describe())

# Create isPayment field
df['isPayment'] = np.where(df['type'].isin(['PAYMENT', 'DEBIT']), 1, 0)

# print(df['isPayment'])

# Create isMovement field
df['isMovement'] = np.where(df['type'].isin(['CASH_OUT', 'TRANSFER']), 1, 0)


# Create accountDiff field
df['accountDiff'] = df['oldbalanceOrg'] - df['oldbalanceDest']


# print(df.head())

# Create features and label variables
features = df[['amount', 'isPayment', 'isMovement', 'accountDiff']]

label = df['isFraud']

# print(features)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.3)

# Normalize the features variables
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fit the model to the training data
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Score the model on the training data
train_score = lr.score(X_train, y_train)
print(f"Your train score is: {train_score}")

# Score the model on the test data
test_score = lr.score(X_test, y_test)
print(f"Your test score is: {test_score}")

y_pred = lr.predict(X_test)

# Print the model coefficients
# print(lr.coef_)

# New transaction data
transaction1 = np.array([123456.78, 0.0, 1.0, 54670.1])
transaction2 = np.array([98765.43, 1.0, 0.0, 8524.75])
transaction3 = np.array([543678.31, 1.0, 0.0, 510025.5])

# Create a new transaction
your_transaction = np.array([1000000.00, 1.0, 0.0, 1000])

# Combine new transactions into a single array
sample_transactions = np.stack([transaction1, transaction2, transaction3, your_transaction])

# Normalize the new transactions
sample_transactions = sc.transform(sample_transactions)

# Predict fraud on the new transactions
print(lr.predict(sample_transactions))

