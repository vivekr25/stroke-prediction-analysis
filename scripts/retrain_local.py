import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib

# Load cleaned and encoded dataset
df = pd.read_csv('data/cleaned_stroke_data.csv')  # replace with correct file if needed

# Define final columns used in app
final_columns = [
    'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
    'gender_Male', 'gender_Other', 'ever_married_Yes',
    'work_type_Never_worked', 'work_type_Private',
    'work_type_Self-employed', 'work_type_children',
    'Residence_type_Urban',
    'smoking_status_formerly smoked', 'smoking_status_never smoked', 'smoking_status_smokes'
]

X = df[final_columns]
y = df['stroke']

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Balance
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_scaled, y)

# Train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_res, y_res)

# Save locally
joblib.dump(model, 'models/random_forest_stroke_model.joblib')
joblib.dump(scaler, 'models/standard_scaler.joblib')

print("✅ Model and scaler re-exported locally.")