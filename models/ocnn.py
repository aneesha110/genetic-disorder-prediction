import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
import pandas as pd
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

# Reshape the data for CNN input (assuming your features are 1D)
X_train_cnn = X_train.reshape((-1, X_train.shape[1], 1, 1))
X_test_cnn = X_test.reshape((-1, X_test.shape[1], 1, 1))

# Flatten the input for StandardScaler
X_train_flat = X_train.reshape((X_train.shape[0], -1))
X_test_flat = X_test.reshape((X_test.shape[0], -1))

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_flat)
X_test_scaled = scaler.transform(X_test_flat)

# Build an improved Convolutional Neural Network (CNN)
model = Sequential()
model.add(Conv2D(64, (3, 1), activation='relu', input_shape=(X_train_cnn.shape[1], 1, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 1), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_cnn, Y_train_1, epochs=20, batch_size=64, validation_split=0.2, verbose=2)

# Evaluate the model on the test set
y_pred_prob = model.predict(X_test_cnn)
y_pred1 = (y_pred_prob > 0.5).astype(int)
# Calculate accuracy
accuracy = accuracy_score(Y_test_1, y_pred1)
print(f"Accuracy: {accuracy * 100:.2f}%")
model = Sequential()
model.add(Conv2D(64, (3, 1), activation='relu', input_shape=(X_train_cnn.shape[1], 1, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 1), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_cnn, Y_train_2, epochs=20, batch_size=64, validation_split=0.2, verbose=2)

# Evaluate the model on the test set
y_pred_prob = model.predict(X_test_cnn)
y_pred2 = (y_pred_prob > 0.5).astype(int)


accuracy = accuracy_score(Y_test_2, y_pred2)
print(f"Accuracy: {accuracy * 100:.2f}%")
