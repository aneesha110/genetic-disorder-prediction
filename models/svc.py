import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
train_data = pd.read_csv('normalized_data.csv')
train_ini = train_data.drop(['Genetic Disorder', 'Disorder Subclass'], axis = 1)
train = train_ini.drop(['Maternal gene','Paternal gene','Mother\'s age','Father\'s age','Heart Rate (rates/min','Birth asphyxia','Folic acid details (peri-conceptional)','White Blood cell count (thousand per microliter)'], axis = 1)
target = train_data[['Genetic Disorder', 'Disorder Subclass']]
target = target.copy()

# Convert "Genetic Disorder" column to category type and encode it
target.loc[:, "Genetic Disorder"] = target.loc[:, "Genetic Disorder"].astype('category').cat.codes

# Convert "Disorder Subclass" column to category type and encode it
target.loc[:, "Disorder Subclass"] = target.loc[:, "Disorder Subclass"].astype('category').cat.codes
X = train.to_numpy()[:23000]
y = target.to_numpy()[:23000]
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

clf1 = SVC(gamma="scale")
clf1.fit(X_train, Y_train_1)

clf2 = SVC(gamma="scale")
clf2.fit(X_train, Y_train_2)

# Evaluate accuracy using the selected features
y_pred1 = clf1.predict(X_test)
accuracy_genetic_algorithm1 = accuracy_score(Y_test_1, y_pred1)

y_pred2 = clf2.predict(X_test)
accuracy_genetic_algorithm2 = accuracy_score(Y_test_2, y_pred2)

print("Accuracy after SVM (Genetic Disorder):", accuracy_genetic_algorithm1)
print("Accuracy after SVM (Genetic Disorder):", accuracy_genetic_algorithm1)