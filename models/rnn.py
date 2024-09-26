import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense

# Load data
train_data = pd.read_csv('train_data.csv')

# Normalization
scaler = MinMaxScaler()
data_norm = pd.DataFrame(scaler.fit_transform(train_data), columns=train_data.columns)

train_ini = data_norm.drop(['Genetic Disorder', 'Disorder Subclass'], axis=1)
train = train_ini.drop(['Mother\'s age','Father\'s age','Blood cell count (mcL)',
                        'Heart Rate (rates/min','Folic acid details (peri-conceptional)',
                        'White Blood cell count (thousand per microliter)'], axis=1)
target = train_data[['Genetic Disorder', 'Disorder Subclass']].copy()

# Convert "Genetic Disorder" column to category type and encode it
target['Genetic Disorder'] = target['Genetic Disorder'].astype('category').cat.codes

# Convert "Disorder Subclass" column to category type and encode it
target['Disorder Subclass'] = target['Disorder Subclass'].astype('category').cat.codes

X = train.values
y = target.values

# Split dataset: 80% for train set and 20% for test set
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape data for RNN input (assuming the shape of input data for RNN)
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# Define RNN model
model1 = Sequential([
    SimpleRNN(64, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(1, activation='sigmoid')  # assuming binary classification
])

model2 = Sequential([
    SimpleRNN(64, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(1, activation='sigmoid')  # assuming binary classification
])

# Compile models
model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train models
model1.fit(X_train, Y_train[:, 0], epochs=10, batch_size=32, validation_split=0.2)
model2.fit(X_train, Y_train[:, 1], epochs=10, batch_size=32, validation_split=0.2)

# Evaluate models
accuracy_genetic_algorithm1 = model1.evaluate(X_test, Y_test[:, 0])[1]
accuracy_genetic_algorithm2 = model2.evaluate(X_test, Y_test[:, 1])[1]

print("Accuracy after RNN (Genetic Disorder):", accuracy_genetic_algorithm1)
print("Accuracy after RNN (Disorder Subclass):", accuracy_genetic_algorithm2)
