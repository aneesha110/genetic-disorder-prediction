from rf1 import rf_classifier_2, rf_classifier_1,  train
import numpy as np
def predict_disorder(rf_classifier_1, rf_classifier_2, selected_features):
    # Dictionary mapping disorder class labels to their names
    disorder_labels = {0: "Mitochondrial genetic inheritance disorder", 1: "Multifactorial genetic inheritance disorder", 2:"Single gene inheritance diseases",3:"No disorder"}  # Update with your actual labels

    # Dictionary mapping disorder subclass labels to their names
    subclass_labels = {0: "Alzheimer's", 1: "Cancer", 2:"Cystic Fibrosis" ,3:"Diabetes", 4:"Hemochromatosis",5:"Lebers Hereditary Optic Neuropathy",6:"Leigh Syndrome",7:"Mitochondrial Myopathy",8:"Tay Sachs",9:"No disorder"}  # Update with your actual subclass labels

    feature_values = []
    for feature_index in selected_features:
        value = float(input(f"Enter value for feature {feature_index}: "))
        feature_values.append(value)

    feature_vector = np.array(feature_values).reshape(1, -1)

    # Predict the disorder class and subclass
    disorder_class = rf_classifier_1.predict(feature_vector)[0]
    disorder_subclass = rf_classifier_2.predict_proba(feature_vector)[0]

    print("Predicted Disorder:", disorder_labels[disorder_class])
    print("Predicted Disorder Subclass:", subclass_labels[np.argmax(disorder_subclass)])

# Example usage:
predict_disorder(rf_classifier_1,rf_classifier_2, train.columns)