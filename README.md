# stroke-prediction-analysis

## 🧠 Stroke Prediction Using Machine Learning

This project builds a machine learning model to predict the likelihood of stroke based on health indicators like age, glucose levels, hypertension, etc.

---

### 📁 Dataset

- **Source**: [Kaggle - Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
- **Rows**: 5,110  
- **Target Variable**: `stroke` (1 = stroke occurred, 0 = no stroke)  
- **Class Imbalance**: Only ~4.9% of records are strokes.

---

### 📊 Exploratory Data Analysis (EDA)

- Checked data types, missing values, and unique values
- Identified **201 missing values** in `bmi`
- **Class imbalance** identified with 95% of patients having no stroke
- Visualizations:
  - Stroke Count Distribution
  - Stroke Distribution by Gender ✅

---

### 🧼 Data Preprocessing (upcoming)

- Handle missing `bmi` values
- Encode categorical features (`gender`, `work_type`, `smoking_status`, etc.)
- Normalize numerical features (e.g., `age`, `avg_glucose_level`, `bmi`)

---

### 🤖 Model Building (coming soon)

We will experiment with:
- Logistic Regression
- Random Forest
- SMOTE to address class imbalance

---

### 🚀 Deployment (later)

- Simple Flask app for real-time prediction
- Publicly hosted via Render or Azure (to be decided)

---

### 📂 Folder Structure

```
project/
│
├── data/                  # Raw data
├── notebooks/             # EDA notebooks
├── scripts/               # Preprocessing and model scripts
├── templates/             # HTML for web app
├── models/                # Trained models (joblib)
├── app.py                 # Flask application
├── README.md              # Project summary
└── requirements.txt       # Dependencies
```

---

### 👤 Author

**Vivek Raghu**  
[LinkedIn](https://www.linkedin.com/in/raghuvivek/)  
GitHub: [vivekr25](https://github.com/vivekr25)
