#  Bank Customer Churn Prediction

A full end-to-end data science portfolio project that predicts bank customer churn using machine learning.
Built as part of my Data Science portfolio.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

---

##  Project Goal

Predict which bank customers are likely to churn (leave the bank), and surface the key drivers behind churn to support data-driven retention decisions.

---

##  Project Structure

```
bank-churn-project/
│
├── data/
│   ├── bank_customer_churn.csv               # Raw dataset (10,000 customers)
│   └── bank_customer_churn_cleaned_data.csv  # Cleaned dataset (Phase 2 output)
│
├── 1_data_cleaning.ipynb          # Phase 2: Data cleaning & validation
├── 2_eda_analysis.ipynb           # Phase 3: Exploratory data analysis
├── 3_feature_engineering.ipynb    # Phase 4: Feature engineering & encoding
├── 4_modelling.ipynb              # Phase 5: Model training & evaluation
│
├── app.py                         # Streamlit deployment app
├── churn_model.pkl                # Trained Gradient Boosting model
├── churn_scaler.pkl               # Fitted StandardScaler
├── requirements.txt               # Python dependencies
└── README.md
```

---

##  Dataset

- **Source:** Bank customer records
- **Size:** 10,000 customers, 12 features
- **Target:** `Exited` — whether a customer churned (1) or stayed (0)
- **Class balance:** 79.6% retained, 20.4% churned

**Features include:** Credit Score, Geography, Age, Tenure, Account Balance, Number of Products, Credit Card status, Active Membership status, Estimated Salary

---

##  Project Phases

### Phase 2:  Data Cleaning
- Handled missing values (median imputation for numeric, mode for categorical)
- Removed 2 duplicate rows
- Fixed decimal ages and incorrect data types
- Final validation: 0 missing values, 0 duplicates, correct dtypes

### Phase 3:  Exploratory Data Analysis
Key findings:
-  **Germany** churn rate is **32.4%** — nearly double France (16.2%) and Spain (16.7%)
-  Customers with **4 products churn at 100%** — cross-selling strategy is broken
-  **Age 45–54** has ~48% churn rate — highest risk demographic
-  Churned customers hold **higher average balances** (£91K vs £73K) — high-value customers leaving
-  **Inactive members** churn at 26.9% vs 14.3% for active members

### Phase 4:  Feature Engineering
Four new features created:
| Feature | Formula | Insight |
|---------|---------|---------|
| `Balance_to_Salary` | Balance / (Salary + 1) | Relative financial commitment to bank |
| `AgeGroup` | pd.cut bins | Makes non-linear age-churn pattern explicit |
| `ZeroBalance` | (Balance == 0).astype(int) | Flags structurally different zero-balance customers |
| `Products_per_Tenure` | NumOfProducts / (Tenure + 1) | Pace of product acquisition |

- One-Hot Encoding applied to Geography
- StandardScaler applied to 7 continuous features

### Phase 5:  Modelling
Three models trained and compared:

| Model | AUC-ROC | F1 | Precision | Recall |
|-------|---------|-----|-----------|--------|
| Logistic Regression | 0.768 | 0.496 | 0.382 | 0.708 |
| Random Forest | 0.845 | 0.562 | 0.804 | 0.432 |
| **Gradient Boosting ** | **0.860** | **0.609** | 0.539 | **0.700** |

**Selected model:** Gradient Boosting Classifier
- Trained on oversampled data (random oversampling on training set only)
- Classification threshold tuned from 0.50 → **0.35** to maximise recall
- 5-fold CV AUC: **0.934 ± 0.007**

### Phase 6: Deployment
Interactive Streamlit app that:
- Accepts customer details as inputs
- Returns churn probability and risk level
- Surfaces key risk signals and protective factors per customer
- Provides tailored retention action recommendations

---

##  Run Locally

```bash
# Clone the repo
git clone https://github.com/your-username/bank-churn-project.git
cd bank-churn-project

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

##  Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Pandas & NumPy | Data manipulation |
| Matplotlib & Seaborn | Visualisation |
| Scikit-learn | Modelling & preprocessing |
| Streamlit | Web app deployment |
| Joblib | Model serialisation |
| Jupyter Notebooks | Analysis & documentation |

---

##  Key Technical Decisions

- **Class imbalance:** Random oversampling applied to training data only — prevents data leakage
- **Encoding:** One-Hot Encoding for Geography — no ordinal assumption, compatible with all model types
- **Scaling:** StandardScaler on continuous features only — binary and encoded columns excluded
- **Threshold:** 0.35 instead of 0.50 — reflects asymmetric cost of missing a churner vs false alarm
- **Metric priority:** AUC-ROC and Recall over accuracy — accuracy is misleading on imbalanced data

---

##  Author

**Lawal Sunmisola**
Data Scientist|Machine Learning Engineer

---

