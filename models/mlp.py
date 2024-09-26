from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
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

# Split data into train and test sets
#X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train MLP model for Genetic Disorder
mlp1 = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)  # You can adjust hidden_layer_sizes and max_iter
mlp1.fit(X_train, Y_train_1)  # Train for Genetic Disorder

# Initialize and train MLP model for Disorder Subclass
mlp2 = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)  # You can adjust hidden_layer_sizes and max_iter
mlp2.fit(X_train, Y_train_2)  # Train for Disorder Subclass

# Predict on test set
y_pred_mlp1 = mlp1.predict(X_test)
y_pred_mlp2 = mlp2.predict(X_test)

# Calculate accuracy
accuracy_mlp1 = accuracy_score(Y_test[:, -2], y_pred_mlp1)
accuracy_mlp2 = accuracy_score(Y_test[:, -1], y_pred_mlp2)

print(f"Accuracy for Genetic Disorder (MLP): {accuracy_mlp1 * 100:.2f}%")
print(f"Accuracy for Disorder Subclass (MLP): {accuracy_mlp2 * 100:.2f}%")
