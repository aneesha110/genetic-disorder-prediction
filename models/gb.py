from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

def train_and_evaluate_gb(X_train, Y_train, X_test, Y_test, imputation_strategy='mean'):
    # Step 1: Impute missing values using SimpleImputer
    imputer = SimpleImputer(strategy=imputation_strategy)
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # Step 2: Define parameter grid for hyperparameter tuning
    param_grid = {
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'max_iter': [50, 100, 200],
    }

    # Step 3: Perform hyperparameter tuning using GridSearchCV
    gb = HistGradientBoostingClassifier()
    grid_search = GridSearchCV(gb, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_imputed, Y_train)

    # Step 4: Train the Gradient Boosting model with the best hyperparameters
    best_params = grid_search.best_params_
    gb_optimized = HistGradientBoostingClassifier(**best_params)
    gb_optimized.fit(X_train_imputed, Y_train)

    # Step 5: Make predictions on the test set
    lib_pred = gb_optimized.predict(X_test_imputed)

    # Step 6: Compute accuracy using accuracy_score
    accuracy = accuracy_score(Y_test, lib_pred)

    return accuracy

# Assuming you have X_train, Y_train_1, X_test, and Y_test_1 defined
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
# Call the function with default parameters
accuracy_optimized_gb_class = train_and_evaluate_gb(X_train, Y_train_1, X_test, Y_test_1)

print(f"Accuracy (Gradient Boosting - Optimized): {accuracy_optimized_gb_class * 100:.2f}%")
accuracy_optimized_gb_subclass = train_and_evaluate_gb(X_train, Y_train_2, X_test, Y_test_2)

print(f"Accuracy (Gradient Boosting - Optimized): {accuracy_optimized_gb_subclass * 100:.2f}%")