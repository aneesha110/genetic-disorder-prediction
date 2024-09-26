from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
train_data = pd.read_csv('normalized_data.csv')
train = train_data.drop(['Genetic Disorder', 'Disorder Subclass'], axis=1)
target = train_data[['Genetic Disorder', 'Disorder Subclass']]

# Convert target columns to category type and encode them
target['Genetic Disorder'] = target['Genetic Disorder'].astype('category').cat.codes
target['Disorder Subclass'] = target['Disorder Subclass'].astype('category').cat.codes

X = train.to_numpy()
y = target.to_numpy()

# Split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.model_selection import GridSearchCV

# Define parameter grid for grid search
param_grid = {
    'n_neighbors': [3, 5, 7, 9],  # You can choose different numbers of neighbors to try
    'weights': ['uniform', 'distance'],  # Different weighting strategies
    'metric': ['euclidean', 'manhattan']  # Different distance metrics
}

# Initialize KNN classifier for Genetic Disorder
knn1 = KNeighborsClassifier()
grid_search1 = GridSearchCV(knn1, param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search1.fit(X_train, Y_train[:, -2])  # Train for Genetic Disorder

# Initialize KNN classifier for Disorder Subclass
knn2 = KNeighborsClassifier()
grid_search2 = GridSearchCV(knn2, param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search2.fit(X_train, Y_train[:, -1])  # Train for Disorder Subclass

# Best parameters and accuracy for Genetic Disorder
print("Best parameters for Genetic Disorder:", grid_search1.best_params_)
print("Best accuracy for Genetic Disorder:", grid_search1.best_score_ * 100)

# Best parameters and accuracy for Disorder Subclass
print("Best parameters for Disorder Subclass:", grid_search2.best_params_)
print("Best accuracy for Disorder Subclass:", grid_search2.best_score_ * 100)

# Predict on test set using best models
best_knn1 = grid_search1.best_estimator_
best_knn2 = grid_search2.best_estimator_

y_pred1 = best_knn1.predict(X_test)
y_pred2 = best_knn2.predict(X_test)

# Calculate accuracy
accuracy1 = accuracy_score(Y_test[:, -2], y_pred1)
accuracy2 = accuracy_score(Y_test[:, -1], y_pred2)

print(f"Accuracy for Genetic Disorder: {accuracy1 * 100:.2f}%")
print(f"Accuracy for Disorder Subclass: {accuracy2 * 100:.2f}%")

