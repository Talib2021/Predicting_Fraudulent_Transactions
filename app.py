import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained model
model = load_model('Model_Fraud.h5')

# Initialize the scaler (same as you used during training)
scaler = StandardScaler()

# Route for the main page
@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the input data from the form
        step = float(request.form['step'])
        
        # Get the type from the dropdown and map to corresponding numeric value
        type_ = request.form['type']
        type_mapping = {
            'Payment': 3,
            'Transfer': 4,
            'Cash_out': 1,
            'Debit': 2
        }
        type_numeric = type_mapping.get(type_, -1)  # Default to -1 if no match

        amount = float(request.form['amount'])
        oldbalanceOrg = float(request.form['oldbalanceOrg'])
        newbalanceOrg = float(request.form['newbalanceOrg'])
        oldbalanceDest = float(request.form['oldbalanceDest'])
        newbalanceDest = float(request.form['newbalanceDest'])
        isFlaggedFraud = float(request.form['isFlaggedFraud'])

        # Preprocess the input data (assuming the same preprocessing as during training)
        input_data = np.array([[step, type_numeric, amount, oldbalanceOrg, newbalanceOrg, oldbalanceDest, newbalanceDest, isFlaggedFraud]])
        
        # Standardize the input data
        input_scaled = scaler.fit_transform(input_data)

        # Make predictions
        prediction = model.predict(input_scaled)
        predicted_class = np.argmax(prediction, axis=1)[0]  # Assuming the model outputs probabilities
        
        # Map predictions to class names (adjust based on your class labels)
        if predicted_class == 0:
            result = "Non-Fraud"
        elif predicted_class == 1:
            result = "Fraud"
        else:
            result = "Unknown"

        return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
