import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

# Load data
train_data = pd.read_csv('normalized_data.csv')

# Normalization
scaler = MinMaxScaler()
data_norm = pd.DataFrame(scaler.fit_transform(train_data), columns=train_data.columns)

train_ini = data_norm.drop(['Genetic Disorder', 'Disorder Subclass'], axis=1)
train = train_ini.drop(['Mother\'s age','Father\'s age','Blood cell count (mcL)','Heart Rate (rates/min','Folic acid details (peri-conceptional)','White Blood cell count (thousand per microliter)'], axis=1)
target = train_data[['Genetic Disorder', 'Disorder Subclass']]
target = target.copy()

# Convert target columns to categorical and encode them
target['Genetic Disorder'] = target['Genetic Disorder'].astype('category').cat.codes
target['Disorder Subclass'] = target['Disorder Subclass'].astype('category').cat.codes

# Splitting data
X = train.to_numpy()[:-20000]
y = target.to_numpy()[:-20000]

# Split dataset: 80% for train set and 20% for test set
num_of_rows = int(len(X) * 0.8)
X_train = X[:num_of_rows]
X_test = X[num_of_rows:]

# Splitting target variables
Y_train = y[:num_of_rows]
Y_train_1 = Y_train[:, -2]  # For Genetic Disorder
Y_train_2 = Y_train[:, -1]  # For Disorder Subclass

Y_test = y[num_of_rows:]
Y_test_1 = Y_test[:, -2]
Y_test_2 = Y_test[:, -1]

# Define parameter grid for grid search
param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.1, 0.01, 0.001],
    'n_estimators': [100, 200, 300]
}

# Grid search for Genetic Disorder classification
clf1 = GridSearchCV(XGBClassifier(random_state=42), param_grid, cv=5)
clf1.fit(X_train, Y_train_1)

# Grid search for Disorder Subclass classification
clf2 = GridSearchCV(XGBClassifier(random_state=42), param_grid, cv=5)
clf2.fit(X_train, Y_train_2)

# Best parameters found by grid search
print("Best parameters for Genetic Disorder classification:", clf1.best_params_)
print("Best parameters for Disorder Subclass classification:", clf2.best_params_)

# Evaluate accuracy using the selected features
y_pred1 = clf1.predict(X_test)
accuracy_genetic_algorithm1 = accuracy_score(Y_test_1, y_pred1)

y_pred2 = clf2.predict(X_test)
accuracy_genetic_algorithm2 = accuracy_score(Y_test_2, y_pred2)

print("Accuracy after Genetic Algorithm (Genetic Disorder):", accuracy_genetic_algorithm1)
print("Accuracy after Genetic Algorithm (Disorder Subclass):", accuracy_genetic_algorithm2)
