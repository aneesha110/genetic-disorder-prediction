import joblib
def predict(data):
    clf1 = joblib.load('rf1.sav')
    clf2 = joblib.load('rf2.sav')
    
    return clf1.predict(data), clf2.predict(data)

