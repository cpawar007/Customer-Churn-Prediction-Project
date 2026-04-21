import streamlit as st
import joblib
import pandas as pd
import os

@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "churn_pred_model.pkl")
    return joblib.load(model_path)

model = load_model()

st.set_page_config(page_title="Churn Predictor", layout="wide")

st.title("📊 Customer Churn Prediction App")
st.markdown("Predict whether a customer is likely to churn based on their details.")

st.sidebar.header("📥 Enter Customer Details")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])

tenure_months = st.sidebar.slider("Tenure (months)", 0, 72, 12)

phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

online_security = st.sidebar.selectbox("Online Security", ["Yes", "No"])
online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No"])
device_protection = st.sidebar.selectbox("Device Protection", ["Yes", "No"])
tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No"])

streaming_tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No"])

contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])

payment_method = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthly_charges = st.sidebar.slider("Monthly Charges", 18, 118, 50)

if st.button("🚀 Predict Churn"):

    with st.spinner("Analyzing customer behavior..."):

        input_data = pd.DataFrame([{
            "gender": gender,
            "senior_citizen": senior_citizen,
            "partner": partner,
            "dependents": dependents,
            "tenure_months": tenure_months,
            "phone_service": phone_service,
            "multiple_lines": multiple_lines,
            "internet_service": internet_service,
            "online_security": online_security,
            "online_backup": online_backup,
            "device_protection": device_protection,
            "tech_support": tech_support,
            "streaming_tv": streaming_tv,
            "streaming_movies": streaming_movies,
            "contract": contract,
            "paperless_billing": paperless_billing,
            "payment_method": payment_method,
            "monthly_charges": monthly_charges
        }])

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

    st.subheader("📈 Prediction Result")

    st.progress(int(probability * 100))

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Churn Probability", f"{probability:.2%}")

    with col2:
        if prediction == 1:
            st.error("⚠️ Customer is likely to churn")
        else:
            st.success("✅ Customer is not likely to churn")

    st.subheader("🧠 Insights")

    if probability > 0.7:
        st.warning("High risk: Consider retention strategies like discounts or support calls.")
    elif probability > 0.4:
        st.info("Moderate risk: Monitor engagement and offer personalized plans.")
    else:
        st.success("Low risk: Customer is stable.")

    with st.expander("📄 View Customer Data"):
        st.dataframe(input_data)
