from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split

train_data = pd.read_csv('normalized_data.csv')
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

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train KNN models
knn1 = KNeighborsClassifier(n_neighbors=5)  # You can choose the number of neighbors
knn1.fit(X_train, Y_train[:, -2])  # Train for Genetic Disorder

knn2 = KNeighborsClassifier(n_neighbors=5)  # You can choose the number of neighbors
knn2.fit(X_train, Y_train[:, -1])  # Train for Disorder Subclass

# Predict on test set
y_pred1 = knn1.predict(X_test)
y_pred2 = knn2.predict(X_test)

# Calculate accuracy
accuracy1 = accuracy_score(Y_test[:, -2], y_pred1)
accuracy2 = accuracy_score(Y_test[:, -1], y_pred2)

print(f"Accuracy for Genetic Disorder: {accuracy1 * 100:.2f}%")
print(f"Accuracy for Disorder Subclass: {accuracy2 * 100:.2f}%")
