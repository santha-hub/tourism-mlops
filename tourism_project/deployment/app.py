"""
Streamlit App – Wellness Tourism Package Prediction
Loads the best model committed by the GitHub Actions pipeline
and serves predictions via an interactive web UI.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Wellness Tourism Package Predictor",
    page_icon="🌿",
    layout="wide"
)

st.title("🌿 Wellness Tourism Package – Purchase Predictor")
st.markdown("""
This app predicts whether a customer is likely to purchase the **Wellness Tourism Package**
based on their profile and interaction data.
""")

# ── Load model ─────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.joblib")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"Model not found. Ensure the pipeline has run and committed the model. Error: {e}")
    st.stop()

# ── Sidebar inputs ─────────────────────────────────────────────────────────────
st.sidebar.header("Customer Details")

age               = st.sidebar.slider("Age", 18, 80, 35)
monthly_income    = st.sidebar.number_input("Monthly Income ($)", 5000, 100000, 20000, 500)
city_tier         = st.sidebar.selectbox("City Tier", [1, 2, 3])
type_of_contact   = st.sidebar.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
occupation        = st.sidebar.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
gender            = st.sidebar.selectbox("Gender", ["Male", "Female"])
marital_status    = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced"])
designation       = st.sidebar.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
passport          = st.sidebar.selectbox("Has Passport?", [0, 1], format_func=lambda x: "Yes" if x else "No")
own_car           = st.sidebar.selectbox("Owns Car?", [0, 1], format_func=lambda x: "Yes" if x else "No")

st.sidebar.header("Interaction Details")

product_pitched         = st.sidebar.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
pitch_satisfaction      = st.sidebar.slider("Pitch Satisfaction Score", 1, 5, 3)
duration_of_pitch       = st.sidebar.slider("Duration of Pitch (mins)", 5, 60, 15)
number_of_followups     = st.sidebar.slider("Number of Follow-ups", 1, 6, 3)
preferred_property_star = st.sidebar.slider("Preferred Property Stars", 1, 5, 3)
number_of_trips         = st.sidebar.slider("Number of Trips/Year", 1, 20, 3)
number_of_persons       = st.sidebar.slider("Persons Visiting", 1, 5, 2)
number_of_children      = st.sidebar.slider("Children Visiting (< 5 yrs)", 0, 3, 0)

# ── Predict ────────────────────────────────────────────────────────────────────
input_data = pd.DataFrame([{
    "Age":                      age,
    "CityTier":                 city_tier,
    "DurationOfPitch":          duration_of_pitch,
    "NumberOfPersonVisiting":   number_of_persons,
    "NumberOfFollowups":        number_of_followups,
    "PreferredPropertyStar":    preferred_property_star,
    "NumberOfTrips":            number_of_trips,
    "Passport":                 passport,
    "PitchSatisfactionScore":   pitch_satisfaction,
    "OwnCar":                   own_car,
    "NumberOfChildrenVisiting": number_of_children,
    "MonthlyIncome":            monthly_income,
    "TypeofContact":            type_of_contact,
    "Occupation":               occupation,
    "Gender":                   gender,
    "ProductPitched":           product_pitched,
    "MaritalStatus":            marital_status,
    "Designation":              designation,
}])

col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer Profile Summary")
    st.dataframe(input_data.T.rename(columns={0: "Value"}), use_container_width=True)

with col2:
    st.subheader("Prediction")
    if st.button("🔮 Predict Purchase Likelihood", type="primary"):
        prediction      = model.predict(input_data)[0]
        probability     = model.predict_proba(input_data)[0]
        purchase_prob   = probability[1] * 100
        no_purchase_prob = probability[0] * 100

        if prediction == 1:
            st.success(f"**Likely to Purchase** the Wellness Tourism Package")
        else:
            st.warning(f"**Unlikely to Purchase** the Wellness Tourism Package")

        st.metric("Purchase Probability",    f"{purchase_prob:.1f}%")
        st.metric("No Purchase Probability", f"{no_purchase_prob:.1f}%")

        st.progress(int(purchase_prob))

        st.markdown("---")
        if prediction == 1:
            st.markdown("**Recommendation:** Contact this customer — high conversion likelihood.")
        else:
            st.markdown("**Recommendation:** Deprioritise — focus resources on higher-probability leads.")

st.markdown("---")
st.caption("Powered by XGBoost + MLflow + GitHub Actions | Visit with Us MLOps Pipeline")
