# 🧠 Stroke Prediction Analysis

A machine learning project to predict the likelihood of stroke using health data. This project showcases end-to-end data science skills from **EDA** to **deployment**, with a focus on real-world healthcare applications.

[![Live Demo](https://img.shields.io/badge/Live%20App-Render-blue)](https://stroke-prediction-analysis.onrender.com/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Source%20Code-green)](https://github.com/vivekr25/stroke-prediction-analysis)

Enter health information such as age, glucose level, BMI, etc., and get an instant stroke risk prediction using a machine learning model trained on real data.
---

## 📚 Table of Contents
- [Dataset](#-dataset)
- [Exploratory Data Analysis](#-exploratory-data-analysis-eda)
- [Smoking Status Analysis](#-stroke-distribution-by-smoking-status)
- [Residence & Work Type Analysis](#-residence--work-type-analysis)
- [Data Preprocessing](#-data-preprocessing)
- [Model Building](#-model-building)
- [Deployment](#-deployment)
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
- Checked column types, missing values, and value counts
- Identified **201 missing values** in the `bmi` column
- Detected significant **class imbalance** (~95% no stroke)
- Created visualizations:
  - Stroke Count Distribution
  - Stroke Distribution by Gender

---

## 🚬 Stroke Distribution by Smoking Status

Analyzed stroke prevalence across:  
**"formerly smoked"**, **"never smoked"**, **"smokes"**, and **"unknown"**.

### 🔍 Observations:
- **"Never smoked"** is the largest group.
- **"Formerly smoked"** shows slightly higher stroke proportion.
- **"Smokes"** has fewer stroke cases than expected.
- **"Unknown"** still includes stroke cases.

💡 **Insight**: Stroke risk spans across all smoking categories — confirming that **stroke is multifactorial**.

---

## 🏘️ Residence & Work Type Analysis

### 🏡 Residence Type:
- **Urban stroke rate**: ~5.2%
- **Rural stroke rate**: ~4.5%
- 💡 Slightly higher in urban areas — possibly due to stress or better diagnosis access.

### 💼 Work Type:
- **Self-employed**: Highest stroke proportion (~7.9%)
- **Private/Govt jobs**: ~5%
- **Children**: Minimal strokes
- 💡 Self-employed individuals may experience more health risks due to irregular schedules or lack of access to care.

---

## 🧼 Data Preprocessing

Steps completed:
- ✅ Filled missing `bmi` values using median imputation
- ✅ Encoded categorical variables using `pd.get_dummies`
- ✅ Normalized numeric features with `StandardScaler`
- ✅ Split dataset into train/test (80/20)

---

## 🤖 Model Building

### ⚠️ Problem:
- **Severe class imbalance** (~95% no-stroke vs ~5% stroke)
- Baseline models had high accuracy but failed to detect stroke cases

---

### 🔄 SMOTE: Synthetic Minority Over-sampling Technique

Used SMOTE to synthetically create more stroke samples in the training set.

### ✅ Result:
- Balanced training set: 50% stroke / 50% no-stroke
- Enabled models to learn patterns in the minority class

---

### 📈 Model Comparison

#### 🔹 Logistic Regression (with SMOTE)
| Metric            | Value     |
|-------------------|-----------|
| Accuracy          | 74.85%    |
| Recall (Stroke=1) | 0.80 ✅   |
| Precision         | 0.14      |
| F1-Score          | 0.24      |

🎯 **Best for real-world screening** where false negatives are costly  
✅ Recommended for healthcare settings

---

#### 🌲 Random Forest (with SMOTE)
| Metric            | Value     |
|-------------------|-----------|
| Accuracy          | 91.78% ✅ |
| Recall (Stroke=1) | 0.14 ❌   |
| Precision         | 0.15      |
| F1-Score          | 0.14      |

⚠️ Prioritizes overall accuracy, but **misses most stroke cases**

---

### 🧠 Takeaway:
- Random Forest has better accuracy
- But **Logistic Regression is better for identifying stroke patients** — critical in healthcare

---

### 💾 Model Saving

Final models were saved using `joblib`:
```python
joblib.dump(rf_smote, 'models/random_forest_stroke_model.joblib')
joblib.dump(scaler, 'models/standard_scaler.joblib')

🚀 Deployment 

## 🚀 Live Demo

The stroke prediction web app is live and can be accessed here:  
🔗 [https://stroke-prediction-analysis.onrender.com](https://stroke-prediction-analysis.onrender.com)

### 🛠️ How to Use
1. Go to the [Live App](https://stroke-prediction-analysis.onrender.com/)
2. Enter health details like age, BMI, glucose levels, etc.
3. Click **"Predict Stroke Risk"**
4. See the result: 🚨 "Stroke Risk" or 😊 "No Stroke"

⚙️ **Custom Threshold**
Lowered the prediction threshold to `0.3` (from default `0.5`) to **prioritize recall** and catch more true stroke cases — important in healthcare where false negatives are riskier than false positives.

### 📈 ROC-AUC Evaluation

To further assess the model's ability to distinguish between stroke and non-stroke cases, we plotted the **Receiver Operating Characteristic (ROC) Curve** and calculated the **Area Under the Curve (AUC)**.

#### ✅ Cross-validated AUC scores (on SMOTE-resampled training data):
[0.9840, 0.9981, 0.9966, 0.9972, 0.9967]
Mean AUC Score: 0.9945

This high score indicates that the model is **very effective** at separating stroke from non-stroke classes in training, thanks to balanced resampling using **SMOTE**.

#### 🧪 ROC Curve on Test Set:
- **AUC = 0.76**
- Shows solid performance on unseen data.
- While lower than the training score (as expected), it still reflects **meaningful predictive power** in identifying stroke risks.

> ℹ️ **Why AUC drops on test data**: SMOTE balances only the training data. The test set retains the original class imbalance, making it harder to predict stroke cases. This drop is natural and reflects real-world conditions.

ROC Curve :

<img src="notebooks/roc_curve.png" width="500"/>


project/
│
├── data/                  # Raw dataset
├── notebooks/             # EDA and modeling notebooks
├── scripts/               # Preprocessing and training scripts
├── templates/             # HTML templates for Flask app
├── models/                # Saved .joblib models
├── app.py                 # Flask backend
├── README.md              # Project summary and documentation
└── requirements.txt       # Python dependencies

---

## 🧪 How to Use

1. Visit the [Live App](https://stroke-prediction-analysis.onrender.com/)
2. Enter patient info into the form
3. Submit to receive real-time stroke risk prediction (based on trained ML model)

---

## ⚙️ Tech Stack

- Python 3.11
- Flask
- Scikit-learn
- SMOTE (Imbalanced-learn)
- Pandas, NumPy, Matplotlib
- Hosted on Render

👤 Author

Vivek Raghunathan
🔗 LinkedIn
💻 GitHub: @vivekr25