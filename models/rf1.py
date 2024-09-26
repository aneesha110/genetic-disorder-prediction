from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
import joblib
# Assuming you have X_train, Y_train_1, X_test, and Y_test_1 defined
train_data = pd.read_csv('train_data.csv')

# Convert target columns to category type and encode them
le = LabelEncoder()
target = train_data[['Genetic Disorder', 'Disorder Subclass']]
target = target.apply(le.fit_transform)

# Split data into train and test sets
X = train_data.drop(['Genetic Disorder', 'Disorder Subclass'], axis=1)
train= train_data.drop(['Genetic Disorder', 'Disorder Subclass'], axis=1)
train = train.drop(['Blood test result','White Blood cell count (thousand per microliter)'], axis=1)
target = train_data[['Genetic Disorder', 'Disorder Subclass']]
target = target.copy()

# Convert "Genetic Disorder" column to category type and encode it
target.loc[:, "Genetic Disorder"] = target.loc[:, "Genetic Disorder"].astype('category').cat.codes

# Convert "Disorder Subclass" column to category type and encode it
target.loc[:, "Disorder Subclass"] = target.loc[:, "Disorder Subclass"].astype('category').cat.codes
X = train.to_numpy()[:-8000]
y = target.to_numpy()[:-8000]

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

# Step 1: Impute missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Step 2: Train the Random Forest model
rf_classifier_1 = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier_1.fit(X_train_imputed, Y_train_1)

# Step 3: Make predictions on the test set
rf_pred = rf_classifier_1.predict(X_test_imputed)

# Step 4: Compute accuracy using accuracy_score
accuracy_default_class = accuracy_score(Y_test_1, rf_pred)

rf_classifier_2 = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier_2.fit(X_train_imputed, Y_train_2)

# Step 3: Make predictions on the test set
rf_pred_2 = rf_classifier_2.predict(X_test_imputed)

# Step 4: Compute accuracy using accuracy_score
accuracy_default_subclass = accuracy_score(Y_test_2, rf_pred_2)

# Call the function with default parameters
print(f"Accuracy (Default): {accuracy_default_class * 100:.2f}%")
print(f"Accuracy (Default): {accuracy_default_subclass * 100:.2f}%")

# Calculate true positives, false positives, and false negatives for the first classifier
tp_rf_1 = sum((rf_pred == 1) & (Y_test_1 == 1))
fp_rf_1 = sum((rf_pred == 1) & (Y_test_1 == 0))
fn_rf_1 = sum((rf_pred == 0) & (Y_test_1 == 1))

# Calculate precision, recall, and F1-score for the first classifier
precision_rf_1 = tp_rf_1 / (tp_rf_1 + fp_rf_1) if (tp_rf_1 + fp_rf_1) > 0 else 0
recall_rf_1 = tp_rf_1 / (tp_rf_1 + fn_rf_1) if (tp_rf_1 + fn_rf_1) > 0 else 0
f1_score_rf_1 = 2 * (precision_rf_1 * recall_rf_1) / (precision_rf_1 + recall_rf_1) if (precision_rf_1 + recall_rf_1) > 0 else 0

# Calculate true positives, false positives, and false negatives for the second classifier
tp_rf_2 = sum((rf_pred_2 == 1) & (Y_test_2 == 1))
fp_rf_2 = sum((rf_pred_2 == 1) & (Y_test_2 == 0))
fn_rf_2 = sum((rf_pred_2 == 0) & (Y_test_2 == 1))

# Calculate precision, recall, and F1-score for the second classifier
precision_rf_2 = tp_rf_2 / (tp_rf_2 + fp_rf_2) if (tp_rf_2 + fp_rf_2) > 0 else 0
recall_rf_2 = tp_rf_2 / (tp_rf_2 + fn_rf_2) if (tp_rf_2 + fn_rf_2) > 0 else 0
f1_score_rf_2 = 2 * (precision_rf_2 * recall_rf_2) / (precision_rf_2 + recall_rf_2) if (precision_rf_2 + recall_rf_2) > 0 else 0

# Print the results
print("Metrics for Genetic Disorder Classifier:")
print("Precision:", precision_rf_1)
print("Recall:", recall_rf_1)
print("F1-score:", f1_score_rf_1)
print("\nMetrics for Disorder Subclass Classifier:")
print("Precision:", precision_rf_2)
print("Recall:", recall_rf_2)
print("F1-score:", f1_score_rf_2)
# save the model to disk
joblib.dump(rf_classifier_1,"rf1.sav") 
joblib.dump(rf_classifier_2,"rf2.sav")            