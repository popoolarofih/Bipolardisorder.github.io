from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from keras.initializers import Orthogonal
import joblib

app = Flask(__name__)

def custom_initializer(config):
    if config['class_name'] == 'Orthogonal':
        return Orthogonal(**config['config'])
    return initializers.get(config)

custom_objects = {'Orthogonal': custom_initializer}
model = load_model('disorder_model.h5', custom_objects=custom_objects)

# Load the scaler
scaler = joblib.load('scaler.pkl')

# Function to preprocess input data
def preprocess_data(year, anxiety, drug_use, depression, disorder):
    # Scale the input data using the loaded scaler
    X = np.array([[year, anxiety, drug_use, depression, disorder]])
    X_scaled = scaler.transform(X)
    return X_scaled

# Function to make predictions
def predict_bipolar_disorder(X):
    prediction = model.predict(X)
    return prediction

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        year = float(request.form['year'])
        anxiety = float(request.form['anxiety'])
        drug_use = float(request.form['drug_use'])
        depression = float(request.form['depression'])
        disorder = float(request.form['disorder'])
        
        # Preprocess the input data
        X = preprocess_data(year, anxiety, drug_use, depression, disorder)
        
        # Make prediction
        prediction = predict_bipolar_disorder(X)
        
        # Render the prediction result template with the prediction
        return render_template('result.html', prediction=prediction[0][0])

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
