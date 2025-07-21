# 🧠 Stroke Prediction Analysis

A machine learning project to predict the likelihood of stroke using health data. This project showcases end-to-end data science skills from **EDA** to **deployment**, with a focus on real-world healthcare applications.

---

## 📚 Table of Contents
- [Dataset](#-dataset)
- [Exploratory Data Analysis](#-exploratory-data-analysis-eda)
- [Smoking Status Analysis](#-stroke-distribution-by-smoking-status)
- [Data Preprocessing](#-data-preprocessing-upcoming)
- [Model Building](#-model-building-coming-soon)
- [Deployment](#-deployment-later)
- [Folder Structure](#-folder-structure)
- [Author](#-author)

---

## 📁 Dataset

- **Source**: [Kaggle - Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
- **Records**: 5,110  
- **Target Variable**: `stroke` (1 = stroke occurred, 0 = no stroke)  
- ⚠️ **Class Imbalance**: Only ~4.9% of patients had a stroke

---

## 📊 Exploratory Data Analysis (EDA)

Initial steps involved:
- Checking column types, missing values, and value counts
- Identified **201 missing values** in the `bmi` column
- Detected significant **class imbalance** (~95% no stroke)
- Created visualizations:
  - Stroke Count Distribution
  - Stroke Distribution by Gender ✅

---

## 🚬 Stroke Distribution by Smoking Status

This chart compares stroke prevalence across smoking categories:  
**"formerly smoked"**, **"never smoked"**, **"smokes"**, and **"unknown"**.

### 🔍 Key Observations:
- **"Never smoked"** group is the largest.
- **"Formerly smoked"** has a **slightly higher proportion of stroke** cases.
- **"Smokes"** group has surprisingly fewer strokes (could be due to age or other health behaviors).
- **"Unknown"** category has notable stroke cases too.

### 💡 Insight:
Stroke cases appear in **all categories**, showing that smoking alone is not a definitive predictor. This reinforces that **stroke risk is multifactorial**.

---

🏘️ Stroke Proportion by Residence Type

This analysis explores whether living in a Rural or Urban area has any correlation with stroke occurrence.

🔍 Key Observations:
	•	Urban residents have a slightly higher stroke rate (~5.2%) compared to rural residents (~4.5%).
	•	The difference is modest but could hint at urban lifestyle stressors or better access to healthcare and diagnosis in cities.

💡 Insight:

While the residence type does not drastically affect stroke likelihood, the slightly elevated urban rate is worth noting. It emphasizes the importance of considering environmental and socioeconomic factors in stroke risk analysis.

------

## 🏢 Stroke Proportion by Work Type

This bar chart compares the proportion of stroke cases across various work types.

### 🔍 Key Observations:
- The **Self-employed** group has the **highest stroke proportion (~7.9%)**, making it the most at-risk category.
- **Private** and **Govt_job** employees follow closely with ~5% stroke proportions.
- The **children** category shows minimal stroke cases, as expected.
- The **Never_worked** category has no reported stroke cases.

### 💡 Insight:
Self-employed individuals may face greater stress, irregular routines, or less access to healthcare, possibly contributing to a higher risk of stroke. This highlights the importance of examining occupational health risks in stroke prevention.

## 🧼 Data Preprocessing (upcoming)

Planned steps:
- Impute missing `bmi` values
- Encode categorical variables: `gender`, `work_type`, `smoking_status`, etc.
- Scale numerical features: `age`, `avg_glucose_level`, `bmi`

---

## 🤖 Model Building (coming soon)

Model experiments will include:
- Logistic Regression
- Random Forest
- SMOTE to address class imbalance

Metrics to evaluate:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix

---

## 🚀 Deployment (later)

A web application using Flask:
- Predict stroke probability in real time
- Hosted publicly via **Render** or **Azure**

---

## 📂 Folder Structure
project/
│
├── data/                  # Raw dataset
├── notebooks/             # Jupyter/Colab notebooks
├── scripts/               # Data preprocessing and ML scripts
├── templates/             # HTML templates for the web app
├── models/                # Saved models (joblib/pickle)
├── app.py                 # Flask application
├── README.md              # Project documentation
└── requirements.txt       # Project dependencies

---


## 👤 Author

**Vivek Raghu**  
[LinkedIn](https://www.linkedin.com/in/raghuvivek/)  
GitHub: [vivekr25](https://github.com/vivekr25)