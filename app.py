from flask import Flask, render_template, request
import joblib
import numpy as np
import os

# Initialize app
app = Flask(__name__)

# Load the model and scaler
model = joblib.load('models/random_forest_stroke_model.joblib')
scaler = joblib.load('models/standard_scaler.joblib')

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # DEBUG: Print form data
        print(request.form)

        # Get input data from form
        features = [
            float(request.form['age']),
            int(request.form['hypertension']),
            int(request.form['heart_disease']),
            float(request.form['avg_glucose_level']),
            float(request.form['bmi']),
            int(request.form['gender_Male']),
            int(request.form['gender_Other']),
            int(request.form['ever_married_Yes']),
            int(request.form['work_type_Never_worked']),
            int(request.form['work_type_Private']),
            int(request.form['work_type_Self-employed']),
            int(request.form['work_type_children']),
            int(request.form['Residence_type_Urban']),
            int(request.form['smoking_status_formerly smoked']),
            int(request.form['smoking_status_never smoked']),
            int(request.form['smoking_status_smokes']),
        ]

        # DEBUGGING
        print("➡️ Feature List Length:", len(features))
        print("➡️ Features:", features)
        sys.stdout.flush()

        # Scale and predict
        input_data = scaler.transform([features])
        prob = model.predict_proba(input_data)[0][1]  # Probability of stroke (class 1)
        prediction = 1 if prob >= 0.3 else 0  # Custom threshold

        result = f"Stroke Risk ⚠️ (Prob: {prob:.2f})" if prediction == 1 else f"No Stroke 😊 (Prob: {prob:.2f})"
        return render_template('index.html', prediction=result)

    except Exception as e:
        return f"Error: {str(e)}"

# For Render deployment
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # use Render's assigned port
    app.run(debug=True, host='0.0.0.0', port=port)