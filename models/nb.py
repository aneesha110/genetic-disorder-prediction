from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load data
train_data = pd.read_csv('normalized_data.csv')
# Split data into train and test sets
X = train_data.drop(['Genetic Disorder', 'Disorder Subclass'], axis=1)
train_ini = train_data.drop(['Genetic Disorder', 'Disorder Subclass'], axis=1)
train = train_ini.drop(['Maternal gene','Mother\'s age','Heart Rate (rates/min','Birth asphyxia','H/O substance abuse','Folic acid details (peri-conceptional)','Assisted conception IVF/ART','Birth defects','Blood test result','White Blood cell count (thousand per microliter)'], axis=1)
target = train_data[['Genetic Disorder', 'Disorder Subclass']]
target = target.copy()

X = train.to_numpy()
y = target.to_numpy()
num_of_rows = int(len(X) * 0.8)
X_train = X[:num_of_rows]
X_test = X[num_of_rows:]

# Splitting target variables
Y_train = y[:num_of_rows]
Y_train_1 = Y_train[:, -2]  # for Genetic Disorder
Y_train_2 = Y_train[:, -1]  # for Disorder Subclass

Y_test = y[num_of_rows:]
Y_test_1 = Y_test[:, -2]
Y_test_2 = Y_test[:, -1]

# Initialize and train MLP model for Genetic Disorder
mlp1 = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
mlp1.fit(X_train, Y_train_1)  # Train for Genetic Disorder

# Initialize and train MLP model for Disorder Subclass
mlp2 = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
mlp2.fit(X_train, Y_train_2)  # Train for Disorder Subclass

# Predict on test set using MLP
y_pred_mlp1 = mlp1.predict(X_test)
y_pred_mlp2 = mlp2.predict(X_test)

# Calculate accuracy using MLP
accuracy_mlp1 = accuracy_score(Y_test_1, y_pred_mlp1)
accuracy_mlp2 = accuracy_score(Y_test_2, y_pred_mlp2)

print(f"Accuracy for Genetic Disorder (MLP): {accuracy_mlp1 * 100:.2f}%")
print(f"Accuracy for Disorder Subclass (MLP): {accuracy_mlp2 * 100:.2f}%")

# Initialize and train Naive Bayes model for Genetic Disorder
nb1 = GaussianNB()
nb1.fit(X_train, Y_train_1)  # Train for Genetic Disorder

# Initialize and train Naive Bayes model for Disorder Subclass
nb2 = GaussianNB()
nb2.fit(X_train, Y_train_2)  # Train for Disorder Subclass

# Predict on test set using Naive Bayes
y_pred_nb1 = nb1.predict(X_test)
y_pred_nb2 = nb2.predict(X_test)

# Calculate accuracy using Naive Bayes
accuracy_nb1 = accuracy_score(Y_test_1, y_pred_nb1)
accuracy_nb2 = accuracy_score(Y_test_2, y_pred_nb2)

print(f"Accuracy for Genetic Disorder (Naive Bayes): {accuracy_nb1 * 100:.2f}%")
print(f"Accuracy for Disorder Subclass (Naive Bayes): {accuracy_nb2 * 100:.2f}%")

# Initialize and train Logistic Regression models
log_reg1 = LogisticRegression(max_iter=1000)  # You can adjust max_iter if needed
log_reg1.fit(X_train, Y_train[:, -2])  # Train for Genetic Disorder

log_reg2 = LogisticRegression(max_iter=1000)  # You can adjust max_iter if needed
log_reg2.fit(X_train, Y_train[:, -1])  # Train for Disorder Subclass

# Predict on test set
y_pred_log_reg1 = log_reg1.predict(X_test)
y_pred_log_reg2 = log_reg2.predict(X_test)

# Calculate accuracy for Logistic Regression
accuracy_log_reg1 = accuracy_score(Y_test[:, -2], y_pred_log_reg1)
accuracy_log_reg2 = accuracy_score(Y_test[:, -1], y_pred_log_reg2)

print(f"Accuracy for Genetic Disorder (Logistic Regression): {accuracy_log_reg1 * 100:.2f}%")
print(f"Accuracy for Disorder Subclass (Logistic Regression): {accuracy_log_reg2 * 100:.2f}%")

