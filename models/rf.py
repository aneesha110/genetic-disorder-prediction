import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
# Load data
train_data = pd.read_csv('normalized_data.csv')

# Convert target columns to category type and encode them
le = LabelEncoder()
target = train_data[['Genetic Disorder', 'Disorder Subclass']]
target = target.apply(le.fit_transform)


# Split data into train and test sets
X = train_data.drop(['Genetic Disorder', 'Disorder Subclass'], axis=1)
train_ini = train_data.drop(['Genetic Disorder', 'Disorder Subclass'], axis=1)
train = train_ini.drop(['Maternal gene','Mother\'s age','Heart Rate (rates/min','Birth asphyxia','H/O substance abuse','Folic acid details (peri-conceptional)','Assisted conception IVF/ART','Birth defects','Blood test result','White Blood cell count (thousand per microliter)'], axis=1)
target = train_data[['Genetic Disorder', 'Disorder Subclass']]
target = target.copy()

# Convert "Genetic Disorder" column to category type and encode it
target.loc[:, "Genetic Disorder"] = target.loc[:, "Genetic Disorder"].astype('category').cat.codes

# Convert "Disorder Subclass" column to category type and encode it
target.loc[:, "Disorder Subclass"] = target.loc[:, "Disorder Subclass"].astype('category').cat.codes
X = train.to_numpy()
y = target.to_numpy()

#split dataset: 80% for train set and 20% for test set
num_of_rows = int(len(X) * 0.8)
X_train = X[:num_of_rows]
X_test = X[num_of_rows:]

#splitting target variables
Y_train = y[:num_of_rows]
Y_train_1 = Y_train[:, -2] #for Genetic Disorder
Y_train_2 = Y_train[:, -1] #for Disorder Subclass

Y_test = y[num_of_rows:]
Y_test_1 = Y_test[:, -2]
Y_test_2 = Y_test[:, -1]


# Define parameter grid for grid search
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize Random Forest classifier for Genetic Disorder
rf1 = RandomForestClassifier(random_state=42)
grid_search1 = GridSearchCV(rf1, param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search1.fit(X_train, Y_train_1)

# Initialize Random Forest classifier for Disorder Subclass
rf2 = RandomForestClassifier(random_state=42)
grid_search2 = GridSearchCV(rf2, param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search2.fit(X_train, Y_train_2)

# Predict on test set using best models
best_rf1 = grid_search1.best_estimator_
best_rf2 = grid_search2.best_estimator_

y_pred1 = best_rf1.predict(X_test)
y_pred2 = best_rf2.predict(X_test)

# Calculate accuracy
accuracy1 = accuracy_score(Y_test[:, -2], y_pred1)
accuracy2 = accuracy_score(Y_test[:, -1], y_pred2)

print(f"Accuracy for Genetic Disorder: {accuracy1 * 100:.2f}%")
print(f"Accuracy for Disorder Subclass: {accuracy2 * 100:.2f}%")