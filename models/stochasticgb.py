import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import SGDClassifier

# Load data
train_data = pd.read_csv('normalized_data.csv')

# Normalization
scaler = MinMaxScaler()
data_norm = pd.DataFrame(scaler.fit_transform(train_data), columns=train_data.columns)

# Drop irrelevant columns
train_ini = data_norm.drop(['Genetic Disorder', 'Disorder Subclass', "Mother's age", "Father's age",
                            "Blood cell count (mcL)", "Heart Rate (rates/min",
                            "Folic acid details (peri-conceptional)",
                            "White Blood cell count (thousand per microliter)"], axis=1)

# Separate features and target variables
X = train_ini.values
target = train_data[['Genetic Disorder', 'Disorder Subclass']]
target['Genetic Disorder'] = target['Genetic Disorder'].astype('category').cat.codes
target['Disorder Subclass'] = target['Disorder Subclass'].astype('category').cat.codes
y = target.values

# Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize SGD classifiers
clf1 = SGDClassifier(loss='log_loss', random_state=42)  # Logistic Regression for Genetic Disorder
clf2 = SGDClassifier(loss='log_loss', random_state=42)  # Logistic Regression for Disorder Subclass

# Fit classifiers
clf1.fit(X_train, Y_train[:, 0])
clf2.fit(X_train, Y_train[:, 1])

# Predictions
y_pred1 = clf1.predict(X_test)
y_pred2 = clf2.predict(X_test)

# Calculate accuracies
accuracy_genetic_disorder = accuracy_score(Y_test[:, 0], y_pred1)
accuracy_disorder_subclass = accuracy_score(Y_test[:, 1], y_pred2)

print("Accuracy after Stochastic Gradient Descent (Genetic Disorder):", accuracy_genetic_disorder)
print("Accuracy after Stochastic Gradient Descent (Disorder Subclass):", accuracy_disorder_subclass)
