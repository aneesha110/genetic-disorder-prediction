import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

train_data = pd.read_csv('train_data.csv')
train_ini = train_data.drop(['Genetic Disorder', 'Disorder Subclass'], axis=1)
train = train_ini.drop(['Maternal gene','Paternal gene','Mother\'s age','Father\'s age','Blood cell count (mcL)','Heart Rate (rates/min','Folic acid details (peri-conceptional)','White Blood cell count (thousand per microliter)'], axis=1)
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
# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape features for CNN input: [samples, features, 1]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build CNN model for Genetic Disorder
model_genetic = Sequential()
model_genetic.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
model_genetic.add(MaxPooling1D(pool_size=2))
model_genetic.add(Flatten())
model_genetic.add(Dense(50, activation='relu'))
model_genetic.add(Dense(1, activation='sigmoid'))
model_genetic.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train CNN model for Genetic Disorder
model_genetic.fit(X_train, Y_train_1, epochs=10, batch_size=32, validation_data=(X_test, Y_test_1))

# Evaluate CNN model for Genetic Disorder
_, accuracy_genetic = model_genetic.evaluate(X_test, Y_test_1)
print("Accuracy after CNN (Genetic Disorder):", accuracy_genetic)

# Build CNN model for Disorder Subclass
model_subclass = Sequential()
model_subclass.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
model_subclass.add(MaxPooling1D(pool_size=2))
model_subclass.add(Flatten())
model_subclass.add(Dense(50, activation='relu'))
model_subclass.add(Dense(1, activation='sigmoid'))
model_subclass.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train CNN model for Disorder Subclass
model_subclass.fit(X_train, Y_train_2, epochs=10, batch_size=32, validation_data=(X_test, Y_test_2))

# Evaluate CNN model for Disorder Subclass
_, accuracy_subclass = model_subclass.evaluate(X_test, Y_test_2)
print("Accuracy after CNN (Disorder Subclass):", accuracy_subclass)
