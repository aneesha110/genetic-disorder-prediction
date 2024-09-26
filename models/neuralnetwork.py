import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd
# Simulate some data for binary classification
# Replace this with your actual data
# Assume X_train, X_test, y_train, y_test are your feature matrices and target vectors
# Here we are using a randomly generated dataset for demonstration purposes

#from sklearn.datasets import make_classification
#X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
train_data = pd.read_csv('train_data.csv')
train = train_data.drop(['Genetic Disorder', 'Disorder Subclass'], axis = 1)
#train = train_ini.drop(['Maternal gene','Paternal gene','Mother\'s age','Father\'s age','Heart Rate (rates/min','Birth asphyxia','Folic acid details (peri-conceptional)','White Blood cell count (thousand per microliter)'], axis = 1)
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

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build a simple neural network
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, Y_train_1, epochs=10, batch_size=32, validation_split=0.2, verbose=2)

# Evaluate the model on the test set
y_pred_prob1 = model.predict(X_test)
y_pred1 = (y_pred_prob1 > 0.5).astype(int)

model.fit(X_train, Y_train_2, epochs=10, batch_size=32, validation_split=0.2, verbose=2)

# Evaluate the model on the test set
y_pred_prob2 = model.predict(X_test)
y_pred2 = (y_pred_prob1 > 0.5).astype(int)
# Calculate accuracy
accuracy1 = accuracy_score(Y_test_1, y_pred1)
accuracy2 = accuracy_score(Y_test_2, y_pred2)
print(f"Accuracy: {accuracy1 * 100:.2f}%")
print(f"Accuracy: {accuracy2 * 100:.2f}%")