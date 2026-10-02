# 🎓 EduGuard AI — Student Academic Risk Prediction System

> **UN SDG 4 — Quality Education** | Machine Learning College Project

EduGuard AI is an early academic risk detection system that identifies students likely to fail **before final grades are issued**, enabling timely educator intervention. It was built with scikit-learn, XGBoost, and Streamlit.

---

## 📂 Project Structure

```
EduGuard-AI/
├── app.py                         # Streamlit web application
├── requirements.txt               # Python dependencies
├── README.md                      # This file
│
├── data/
│   └── student-mat.csv            # ← YOU MUST PLACE THIS FILE HERE
│
├── models/
│   ├── academic_risk_model.pkl    # Saved model bundle (created after training)
│   └── model_metadata.json        # Human-readable metrics (created after training)
│
└── src/
    ├── data_preprocessing.py      # Dataset loading, cleaning, feature engineering
    ├── train_model.py             # Model training, evaluation, persistence
    ├── evaluate_model.py          # Evaluation utilities for the Streamlit dashboard
    └── prediction.py              # Prediction interface for the Streamlit app
```

---

## ⚙️ Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the dataset

Download **student-mat.csv** from the UCI Machine Learning Repository:

> **https://archive.ics.uci.edu/dataset/320/student+performance**

- Click **Download** on that page to get `student+performance.zip`
- Extract the archive
- Copy `student-mat.csv` into the `data/` directory

```
EduGuard-AI/
└── data/
    └── student-mat.csv   ← place it here
```

### 3. Train the models

```bash
python src/train_model.py
```

This will:
- Load and preprocess `student-mat.csv`
- Train Logistic Regression, Random Forest, and XGBoost
- Evaluate each model on a held-out test set (80/20 stratified split)
- Select the best model by F1-Score
- Save `models/academic_risk_model.pkl` and `models/model_metadata.json`

### 4. Launch the Streamlit app

```bash
streamlit run app.py
```

Open the URL displayed in the terminal (default: http://localhost:8501)

---

## 🎯 Target Variable

| Condition   | Label         | Class |
|-------------|---------------|-------|
| G3 < 10     | High Risk     | 1     |
| G3 ≥ 10    | Lower Risk    | 0     |

> **Important**: G1, G2, and G3 are **completely excluded** from the model's input features to prevent target leakage. The model uses only demographic, social, and behavioural attributes.

---

## 🤖 Models Trained

| Model                | Notes                                    |
|----------------------|------------------------------------------|
| Logistic Regression  | Baseline linear classifier               |
| Random Forest        | 200 trees, balanced class weights        |
| XGBoost              | Gradient boosting, logloss eval metric   |

The best model (by F1-Score) is automatically selected and saved.

---

## 📊 Evaluation Metrics

All metrics are computed on the real held-out test set:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **ROC-AUC**
- **Confusion Matrix**
- **Classification Report**

---

## 🖥️ Streamlit App Pages

| Page                    | Description                                             |
|-------------------------|---------------------------------------------------------|
| 🏠 Home                  | Overview, quick stats, getting started guide            |
| 📊 Data Analysis         | Interactive Plotly charts exploring the dataset         |
| 🤖 Model Performance     | Side-by-side comparison of all three models             |
| 🔮 Student Risk Prediction | Input form → instant risk prediction with confidence  |
| 💡 Feature Insights      | Feature importance and per-feature violin plots         |
| 🌍 About / SDG 4         | Project background, SDG 4 alignment, ethical notes      |

---

## 📐 Features Used by the Model (30 total)

| Feature      | Description                        |
|--------------|------------------------------------|
| school       | Student's school (GP / MS)         |
| sex          | Student's sex                      |
| age          | Age (15–22)                        |
| address      | Urban or Rural                     |
| famsize      | Family size (≤3 or >3)            |
| Pstatus      | Parents' cohabitation status       |
| Medu         | Mother's education level (0–4)     |
| Fedu         | Father's education level (0–4)     |
| Mjob         | Mother's job                       |
| Fjob         | Father's job                       |
| reason       | Reason for choosing this school    |
| guardian     | Student's guardian                 |
| traveltime   | Travel time to school              |
| studytime    | Weekly study time                  |
| failures     | Number of past class failures      |
| schoolsup    | Extra educational support (school) |
| famsup       | Family educational support         |
| paid         | Extra paid classes                 |
| activities   | Extra-curricular activities        |
| nursery      | Attended nursery school            |
| higher       | Wants higher education             |
| internet     | Internet access at home            |
| romantic     | In a romantic relationship         |
| famrel       | Family relationship quality (1–5)  |
| freetime     | Free time after school (1–5)       |
| goout        | Going out with friends (1–5)       |
| Dalc         | Workday alcohol consumption (1–5)  |
| Walc         | Weekend alcohol consumption (1–5)  |
| health       | Current health status (1–5)        |
| absences     | Number of school absences          |

> **Excluded from model**: G1 (1st period grade), G2 (2nd period grade), G3 (final grade — used only to derive the target label)

---

## 🌍 SDG 4 — Quality Education

EduGuard AI aligns with **UN Sustainable Development Goal 4**:

> *"Ensure inclusive and equitable quality education and promote lifelong learning opportunities for all."*

Specifically:
- **4.1** — Supporting completion of quality secondary education
- **4.5** — Identifying vulnerable students to reduce educational disparities
- **4.a** — Demonstrating AI-powered tools to improve educational environments

---

## ⚠️ Ethical Notes

- No target leakage — G1/G2/G3 are never used as input features
- Class imbalance handled with `class_weight="balanced"` in applicable models
- Feature importance is exposed for model transparency
- Predictions should supplement, not replace, educator judgement

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **scikit-learn** — Logistic Regression, Random Forest, preprocessing
- **XGBoost** — Gradient boosting classifier
- **Streamlit** — Web application framework
- **Plotly** — Interactive data visualisations
- **pandas / NumPy** — Data manipulation
- **joblib** — Model serialisation
