import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier


train_data = pd.read_csv('train_data.csv')

#normalization
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data_norm = pd.DataFrame(scaler.fit_transform(train_data), columns=train_data.columns)

# Split data into train and test sets
X = train_data.drop(['Genetic Disorder', 'Disorder Subclass'], axis=1)
train_ini = train_data.drop(['Genetic Disorder', 'Disorder Subclass'], axis=1)
train = train_ini.drop(['Patient Age','Genes in mother\'s side','Inherited from father','Paternal gene','Maternal gene','Respiratory Rate (breaths/min)','Heart Rate (rates/min','H/O substance abuse','Folic acid details (peri-conceptional)','Birth defects','Blood test result','White Blood cell count (thousand per microliter)'], axis=1)
target = train_data[['Genetic Disorder', 'Disorder Subclass']]
target = target.copy()

# Convert "Genetic Disorder" column to category type and encode it
target.loc[:, "Genetic Disorder"] = target.loc[:, "Genetic Disorder"].astype('category').cat.codes

# Convert "Disorder Subclass" column to category type and encode it
target.loc[:, "Disorder Subclass"] = target.loc[:, "Disorder Subclass"].astype('category').cat.codes
# Splitting data
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

print("Unique classes in Y_train_1:", np.unique(Y_train_1))
print("Unique classes in Y_train_2:", np.unique(Y_train_2))

clf1 = XGBClassifier(random_state=42)
clf1.fit(X_train, Y_train_1)

clf2 = XGBClassifier(random_state=42)
clf2.fit(X_train, Y_train_2)

# Evaluate accuracy using the selected features
y_pred1 = clf1.predict(X_test)
accuracy_genetic_algorithm1 = accuracy_score(Y_test_1, y_pred1)

y_pred2 = clf2.predict(X_test)
accuracy_genetic_algorithm2 = accuracy_score(Y_test_2, y_pred2)

print("Accuracy after Genetic Algorithm (Genetic Disorder):", accuracy_genetic_algorithm1)
print("Accuracy after Genetic Algorithm (Disorder Subclass):", accuracy_genetic_algorithm2)
