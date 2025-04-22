from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
import os
import joblib
import pandas as pd
import numpy as np

data = {
    'Day': ['Sunny', 'Windy', 'Sunny', 'Windy', 'Windy', 'Sunny', 'Windy', 'Sunny', 'Sunny'], 
    'Temperature': ['Cool', 'Cool', 'Hot', 'Hot', 'Hot', 'Cool', 'Hot', 'Hot', 'Hot'], 
    'Class': ['Play', 'Not Play', 'Not Play', 'Play', 'Play', 'Play', 'Not Play', 'Play', 'Play']
}

df = pd.DataFrame(data)
x_raw = df[['Day','Temperature']]
y_raw = df['Class']

onehot_encoder = OneHotEncoder()
label_encoder = LabelEncoder()

x_encoded = onehot_encoder.fit_transform(x_raw).toarray()
y_encoded = label_encoder.fit_transform(y_raw)

x_train,x_test,y_train,y_test = train_test_split(x_encoded,y_encoded,test_size=0.3)

model = RandomForestClassifier()
model.fit(x_train,y_train)

joblib.dump(model,'model.pkl')
joblib.dump(onehot_encoder,'onehot.pkl')
joblib.dump(label_encoder,'label.pkl')