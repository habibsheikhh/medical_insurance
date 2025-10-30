from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('premium_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

def classify_bmi(bmi):
    if bmi < 18.5:
        return 'Under Weight'
    elif bmi < 25:
        return 'Normal'
    elif bmi < 30:
        return 'Over Weight'
    else:
        return 'Obecity'

@app.route('/predict', methods=['POST'])
def predict():
    # Extract values
    data = [float(x) for x in request.form.values()]
    
    # Assign variables for readability
    Age, Diabetes, BP, Transplants, Chronic, Height, Weight, Allergies, Cancer, Surgeries = data

    # Compute BMI
    bmi = Weight / ((Height / 100) ** 2)
    bmi_category = classify_bmi(bmi)

    # Initialize BMI one-hot columns
    Normal = Obecity = OverWeight = UnderWeight = 0
    if bmi_category == 'Normal':
        Normal = 1
    elif bmi_category == 'Obecity':
        Obecity = 1
    elif bmi_category == 'Over Weight':
        OverWeight = 1
    else:
        UnderWeight = 1

    # Create feature array matching model training order
    features = np.array([[Age, Diabetes, BP, Transplants, Chronic, Height, Weight,
                          Allergies, Cancer, Surgeries, Normal, Obecity, OverWeight, UnderWeight]])

    # Scale features
    features = scaler.transform(features)

    # Predict
    prediction = model.predict(features)[0]
    
    return render_template('index.html',
                           prediction_text=f'Predicted Premium Price: ₹{prediction:.2f}')

if __name__ == "__main__":
    app.run(debug=True)
