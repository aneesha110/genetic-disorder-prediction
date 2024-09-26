import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# Load data
train_data = pd.read_csv('normalized_data.csv')

# Drop irrelevant columns
train = train_data.drop(['Genetic Disorder', 'Disorder Subclass', 'Maternal gene', 'Paternal gene', 
                         "Mother's age", "Father's age", "Heart Rate (rates/min", "Birth asphyxia", 
                         "Folic acid details (peri-conceptional)", 
                         "White Blood cell count (thousand per microliter)"], axis=1)

# Encode target variables
label_encoder = LabelEncoder()
train_data['Genetic Disorder'] = label_encoder.fit_transform(train_data['Genetic Disorder'])
train_data['Disorder Subclass'] = label_encoder.fit_transform(train_data['Disorder Subclass'])

# Split features and target variables
X = train.values
y_genetic_disorder = train_data['Genetic Disorder'].values
y_disorder_subclass = train_data['Disorder Subclass'].values

# Split dataset into train and test sets
X_train, X_test, y_train_genetic, y_test_genetic, y_train_subclass, y_test_subclass = train_test_split(X, y_genetic_disorder, y_disorder_subclass, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape features for LSTM input: [samples, timesteps, features]
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# Build LSTM model for Genetic Disorder
model_genetic = Sequential()
model_genetic.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model_genetic.add(Dropout(0.2))
model_genetic.add(LSTM(units=50))
model_genetic.add(Dropout(0.2))
model_genetic.add(Dense(1, activation='sigmoid'))
model_genetic.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train LSTM model for Genetic Disorder
model_genetic.fit(X_train, y_train_genetic, epochs=10, batch_size=32, validation_data=(X_test, y_test_genetic))

# Evaluate LSTM model for Genetic Disorder
_, accuracy_genetic = model_genetic.evaluate(X_test, y_test_genetic)
print("Accuracy after LSTM (Genetic Disorder):", accuracy_genetic)

# Build LSTM model for Disorder Subclass
model_subclass = Sequential()
model_subclass.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model_subclass.add(Dropout(0.2))
model_subclass.add(LSTM(units=50))
model_subclass.add(Dropout(0.2))
model_subclass.add(Dense(1, activation='sigmoid'))
model_subclass.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train LSTM model for Disorder Subclass
model_subclass.fit(X_train, y_train_subclass, epochs=10, batch_size=32, validation_data=(X_test, y_test_subclass))

# Evaluate LSTM model for Disorder Subclass
_, accuracy_subclass = model_subclass.evaluate(X_test, y_test_subclass)
print("Accuracy after LSTM (Disorder Subclass):", accuracy_subclass)
