import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem; font-weight: 700;
        color: #1A237E; margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1rem; color: #546E7A; margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8F9FA; border-radius: 12px;
        padding: 1.2rem; text-align: center;
        border: 1px solid #E0E0E0;
    }
    .churn-high {
        background: linear-gradient(135deg, #FFEBEE, #FFCDD2);
        border: 2px solid #F44336; border-radius: 16px; padding: 1.5rem;
    }
    .churn-low {
        background: linear-gradient(135deg, #E3F2FD, #BBDEFB);
        border: 2px solid #2196F3; border-radius: 16px; padding: 1.5rem;
    }
    .churn-medium {
        background: linear-gradient(135deg, #FFF8E1, #FFECB3);
        border: 2px solid #FF9800; border-radius: 16px; padding: 1.5rem;
    }
    .insight-box {
        background: #F3F4F6; border-left: 4px solid #1A237E;
        padding: 0.8rem 1rem; border-radius: 0 8px 8px 0;
        margin: 0.4rem 0; font-size: 0.9rem;
    }
    .section-header {
        font-size: 1.1rem; font-weight: 600;
        color: #1A237E; margin: 1rem 0 0.5rem 0;
    }
    div[data-testid="stSidebar"] { background-color: #F0F4FF; }
</style>
""", unsafe_allow_html=True)


# ── Load model & scaler ───────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model  = joblib.load("churn_model.pkl")
    scaler = joblib.load("churn_scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

THRESHOLD     = 0.35
NUM_TO_SCALE  = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary',
                 'Balance_to_Salary', 'Products_per_Tenure']
FEATURE_COLS  = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
                 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
                 'Balance_to_Salary', 'ZeroBalance', 'Products_per_Tenure',
                 'Geo_France', 'Geo_Germany', 'Geo_Spain']


# ── Prediction function ───────────────────────────────────────
def predict_churn(credit_score, age, tenure, balance, num_products,
                  has_cr_card, is_active, salary, geography):

    bal_to_sal   = balance / (salary + 1)
    zero_balance = 1 if balance == 0 else 0
    prod_per_ten = num_products / (tenure + 1)

    raw = pd.DataFrame([{
        'CreditScore':       credit_score,
        'Age':               age,
        'Tenure':            tenure,
        'Balance':           balance,
        'EstimatedSalary':   salary,
        'Balance_to_Salary': bal_to_sal,
        'Products_per_Tenure': prod_per_ten,
    }])
    raw[NUM_TO_SCALE] = scaler.transform(raw[NUM_TO_SCALE])

    row = {col: 0 for col in FEATURE_COLS}
    row['CreditScore']        = raw['CreditScore'].values[0]
    row['Age']                = raw['Age'].values[0]
    row['Tenure']             = raw['Tenure'].values[0]
    row['Balance']            = raw['Balance'].values[0]
    row['NumOfProducts']      = num_products
    row['HasCrCard']          = has_cr_card
    row['IsActiveMember']     = is_active
    row['EstimatedSalary']    = raw['EstimatedSalary'].values[0]
    row['Balance_to_Salary']  = raw['Balance_to_Salary'].values[0]
    row['ZeroBalance']        = zero_balance
    row['Products_per_Tenure'] = raw['Products_per_Tenure'].values[0]
    row[f'Geo_{geography}']   = 1

    X   = pd.DataFrame([row])[FEATURE_COLS]
    prob = model.predict_proba(X)[0][1]
    pred = int(prob >= THRESHOLD)
    return prob, pred


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 Customer Churn Predictor")
    st.markdown("*Bank Customer Retention Tool*")
    st.divider()

    st.markdown("### 👤 Customer Demographics")
    age       = st.slider("Age", 18, 92, 38)
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])

    st.markdown("### 💳 Account Information")
    credit_score = st.slider("Credit Score", 350, 850, 650)
    tenure       = st.slider("Tenure (Years)", 0, 10, 5)
    balance      = st.number_input("Account Balance (£)", 0.0, 260000.0, 75000.0, step=1000.0)
    salary       = st.number_input("Estimated Salary (£)", 10000.0, 200000.0, 100000.0, step=1000.0)

    st.markdown("### 📦 Product & Engagement")
    num_products = st.selectbox("Number of Products", [1, 2, 3, 4], index=1)
    has_cr_card  = st.radio("Has Credit Card?", ["Yes", "No"], horizontal=True)
    is_active    = st.radio("Active Member?",   ["Yes", "No"], horizontal=True)

    has_cr_card_val = 1 if has_cr_card == "Yes" else 0
    is_active_val   = 1 if is_active   == "Yes" else 0

    st.divider()
    predict_btn = st.button("🔍 Predict Churn Risk", use_container_width=True, type="primary")


# ── Main panel ────────────────────────────────────────────────
st.markdown('<div class="main-title">🏦 Bank Customer Churn Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter customer details in the sidebar and click Predict to assess churn risk</div>', unsafe_allow_html=True)

# Model info strip
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="metric-card"><div style="font-size:1.5rem">0.860</div><div style="color:#546E7A;font-size:0.8rem">Model AUC-ROC</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="metric-card"><div style="font-size:1.5rem">0.700</div><div style="color:#546E7A;font-size:0.8rem">Recall (Churn)</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="metric-card"><div style="font-size:1.5rem">0.609</div><div style="color:#546E7A;font-size:0.8rem">F1 Score</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="metric-card"><div style="font-size:1.5rem">10,000</div><div style="color:#546E7A;font-size:0.8rem">Training Customers</div></div>', unsafe_allow_html=True)

st.markdown("---")

if predict_btn:
    prob, pred = predict_churn(
        credit_score, age, tenure, balance,
        num_products, has_cr_card_val, is_active_val,
        salary, geography
    )
    pct = prob * 100

    # ── Result card ───────────────────────────────────────────
    col_res, col_detail = st.columns([1, 1], gap="large")

    with col_res:
        if pct >= 60:
            css_class = "churn-high"
            emoji     = "🔴"
            verdict   = "HIGH CHURN RISK"
            color     = "#C62828"
            action    = "Immediate retention intervention recommended"
        elif pct >= 35:
            css_class = "churn-medium"
            emoji     = "🟠"
            verdict   = "MODERATE CHURN RISK"
            color     = "#E65100"
            action    = "Proactive engagement advised"
        else:
            css_class = "churn-low"
            emoji     = "🟢"
            verdict   = "LOW CHURN RISK"
            color     = "#1565C0"
            action    = "Monitor with standard retention practices"

        st.markdown(f"""
        <div class="{css_class}">
            <div style="font-size:2.8rem; text-align:center">{emoji}</div>
            <div style="font-size:1.6rem; font-weight:700; color:{color}; text-align:center">
                {verdict}
            </div>
            <div style="font-size:3rem; font-weight:800; color:{color}; text-align:center">
                {pct:.1f}%
            </div>
            <div style="text-align:center; color:#546E7A; font-size:0.9rem">
                Churn Probability
            </div>
            <hr style="border-color:{color}; opacity:0.3">
            <div style="text-align:center; font-size:0.95rem; color:#424242">
                {action}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Probability bar
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Risk Gauge**")
        st.progress(min(int(pct), 100))
        col_l, col_r = st.columns(2)
        col_l.caption("0% — No Risk")
        col_r.caption("100% — Certain Churn")

    with col_detail:
        st.markdown('<div class="section-header">📋 Customer Profile Summary</div>', unsafe_allow_html=True)

        profile_data = {
            "Age":            age,
            "Geography":      geography,
            "Credit Score":   credit_score,
            "Tenure":         f"{tenure} years",
            "Balance":        f"£{balance:,.0f}",
            "Salary":         f"£{salary:,.0f}",
            "Products":       num_products,
            "Credit Card":    has_cr_card,
            "Active Member":  is_active,
        }
        for k, v in profile_data.items():
            st.markdown(f"**{k}:** {v}")

        st.markdown('<div class="section-header">🔍 Key Risk Factors Detected</div>', unsafe_allow_html=True)

        risk_flags = []
        safe_flags = []

        if age >= 45:
            risk_flags.append(f"Age {age} — customers aged 45–64 churn at ~48%")
        if num_products >= 3:
            risk_flags.append(f"{num_products} products — churn rate is {83 if num_products==3 else 100}%")
        if geography == "Germany":
            risk_flags.append("Germany market — 32.4% churn rate vs 16% elsewhere")
        if is_active_val == 0:
            risk_flags.append("Inactive member — inactive customers churn at 26.9%")
        if balance > 80000:
            risk_flags.append(f"Balance £{balance:,.0f} — high-balance customers at greater risk")

        if is_active_val == 1:
            safe_flags.append("Active member — engagement reduces churn risk")
        if num_products == 2:
            safe_flags.append("2 products — lowest churn rate segment (7.6%)")
        if age < 35:
            safe_flags.append(f"Age {age} — younger customers churn at only ~8–10%")
        if balance == 0:
            safe_flags.append("Zero balance — lower churn risk segment (13.8%)")

        if risk_flags:
            st.markdown("**⚠️ Risk signals:**")
            for flag in risk_flags:
                st.markdown(f'<div class="insight-box">⚠️ {flag}</div>', unsafe_allow_html=True)

        if safe_flags:
            st.markdown("**✅ Protective factors:**")
            for flag in safe_flags:
                st.markdown(f'<div class="insight-box" style="border-color:#2196F3">✅ {flag}</div>', unsafe_allow_html=True)

        if not risk_flags and not safe_flags:
            st.info("No strong risk or protective signals detected for this profile.")

    # ── Recommended Actions ───────────────────────────────────
    st.markdown("---")
    st.markdown("### 💡 Recommended Retention Actions")

    rec_col1, rec_col2, rec_col3 = st.columns(3)

    with rec_col1:
        st.markdown("**🎯 Immediate**")
        if pct >= 60:
            st.markdown("- Assign dedicated relationship manager")
            st.markdown("- Offer personalised retention package")
            st.markdown("- Schedule proactive check-in call")
        elif pct >= 35:
            st.markdown("- Send personalised engagement email")
            st.markdown("- Offer relevant product upgrade")
            st.markdown("- Re-activate via app notification")
        else:
            st.markdown("- Maintain standard service quality")
            st.markdown("- Include in quarterly NPS survey")

    with rec_col2:
        st.markdown("**📦 Product Actions**")
        if num_products >= 3:
            st.markdown("- Review product fit — possible over-selling")
            st.markdown("- Consider product simplification offer")
        elif num_products == 1:
            st.markdown("- Identify relevant second product")
            st.markdown("- Targeted cross-sell at next interaction")
        else:
            st.markdown("- Current product portfolio is optimal")
            st.markdown("- Monitor satisfaction with existing products")

    with rec_col3:
        st.markdown("**📍 Geographic Focus**")
        if geography == "Germany":
            st.markdown("- Escalate to Germany retention team")
            st.markdown("- Apply local market retention offer")
            st.markdown("- Review German product pricing")
        else:
            st.markdown("- Standard regional retention protocol")
            st.markdown("- No geographic escalation required")

else:
    # Default state — project overview
    st.markdown("### 📊 About This Model")

    tab1, tab2, tab3 = st.tabs(["Project Overview", "Model Performance", "Key EDA Findings"])

    with tab1:
        st.markdown("""
        This application is the deployment phase of a full end-to-end data science project
        built to predict bank customer churn.

        **Project Pipeline:**
        - **Phase 1** — Data Cleaning: handled missing values, duplicates, type errors
        - **Phase 2** — EDA: uncovered key churn drivers across 10,000 customers
        - **Phase 3** — Feature Engineering: created 4 new predictive features
        - **Phase 4** — Modelling: trained and compared 3 ML models
        - **Phase 5** — Deployment: this Streamlit app

        **Final Model:** Gradient Boosting Classifier trained on oversampled data with
        classification threshold optimised to 0.35 for maximum recall.
        """)

    with tab2:
        perf_df = pd.DataFrame({
            'Model':     ['Logistic Regression', 'Random Forest', 'Gradient Boosting ✅'],
            'AUC-ROC':   [0.768, 0.845, 0.860],
            'F1 Score':  [0.496, 0.562, 0.609],
            'Precision': [0.382, 0.804, 0.539],
            'Recall':    [0.708, 0.432, 0.700],
        })
        st.dataframe(perf_df, use_container_width=True, hide_index=True)
        st.caption("✅ Gradient Boosting selected for deployment based on highest AUC-ROC, F1, and strong Recall")

    with tab3:
        findings = [
            ("🌍 Germany", "32.4% churn rate — nearly double France (16.2%) and Spain (16.7%)"),
            ("📦 4 Products", "100% churn rate — cross-selling strategy needs urgent review"),
            ("👴 Age 45–54", "~48% churn rate — highest risk demographic"),
            ("💰 High Balance", "Churned customers hold £91K avg vs £73K for retained"),
            ("💤 Inactive Members", "26.9% churn vs 14.3% for active — engagement is protective"),
            ("💳 Credit Score", "Near-identical between groups — not a churn predictor"),
        ]
        for title, desc in findings:
            st.markdown(f"**{title}** — {desc}")

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.caption("Built by Lawal Sunmisola Barakat · Bank Customer Churn Prediction · Data Science Portfolio Project · 2025")
