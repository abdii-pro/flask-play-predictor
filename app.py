from flask import Flask,render_template,request
import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        day = request.form['day']
        temperature = request.form['temperature']

        model = joblib.load('model.pkl')
        onehot = joblib.load('onehot.pkl')
        label = joblib.load('label.pkl')

        new_data = pd.DataFrame([[day,temperature]],columns=['Day','Temperature'])
        encoded_new_data = onehot.transform(new_data).toarray()

        prediction = model.predict(encoded_new_data)
        result = label.inverse_transform(prediction)[0]
        
        return render_template('index.html', prediction_text = f"Prediction: {result}")

if __name__ == "__main__":
    app.run(debug=True)