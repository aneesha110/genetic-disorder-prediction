import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense

# Load data
train_data = pd.read_csv('normalized_data.csv')

# Normalization
scaler = MinMaxScaler()
data_norm = pd.DataFrame(scaler.fit_transform(train_data), columns=train_data.columns)

# Split data into train and test sets
X = train_data.drop(['Genetic Disorder', 'Disorder Subclass'], axis=1)
train_ini = train_data.drop(['Genetic Disorder', 'Disorder Subclass'], axis=1)
train = train_ini.drop(['Maternal gene','Mother\'s age','Heart Rate (rates/min','Birth asphyxia','H/O substance abuse','Folic acid details (peri-conceptional)','Assisted conception IVF/ART','Birth defects','Blood test result','White Blood cell count (thousand per microliter)'], axis=1)
target = train_data[['Genetic Disorder', 'Disorder Subclass']]
target = target.copy()

# Convert "Genetic Disorder" column to category type and encode it
target['Genetic Disorder'] = target['Genetic Disorder'].astype('category').cat.codes

# Convert "Disorder Subclass" column to category type and encode it
target['Disorder Subclass'] = target['Disorder Subclass'].astype('category').cat.codes

X = train.values[:-20000]
y = target.values[:-20000]

# Split dataset: 80% for train set and 20% for test set
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define ANN model for Genetic Disorder
model1 = Sequential([
    Dense(64, input_shape=(X_train.shape[1],), activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Define ANN model for Disorder Subclass
model2 = Sequential([
    Dense(64, input_shape=(X_train.shape[1],), activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
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

print("Accuracy after ANN (Genetic Disorder):", accuracy_genetic_algorithm1)
print("Accuracy after ANN (Disorder Subclass):", accuracy_genetic_algorithm2)
