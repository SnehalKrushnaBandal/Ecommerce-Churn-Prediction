import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="E-Commerce Customer Churn Prediction",
    page_icon="🛒",
    layout="wide"
)

# -----------------------------
# Load Trained Model
# -----------------------------
model = joblib.load("models/churn_model.pkl")

# -----------------------------
# Application Title
# -----------------------------
st.title("🛒 E-Commerce Customer Churn Prediction")

st.write(
    "This application predicts whether an e-commerce customer "
    "is likely to churn based on their behavioral and demographic information."
)

st.info(
    "Enter the customer's details below and click **Predict Churn**."
)

# -----------------------------
# Customer Information
# -----------------------------

st.header("👤 Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    tenure = st.number_input(
        "Tenure (months)",
        min_value=0.0,
        max_value=100.0,
        value=10.0
    )

    preferred_login_device = st.selectbox(
        "Preferred Login Device",
        ["Mobile Phone", "Computer", "Phone"]
    )

    city_tier = st.selectbox(
        "City Tier",
        [1, 2, 3]
    )

    warehouse_to_home = st.number_input(
        "Warehouse to Home (km)",
        min_value=0.0,
        max_value=200.0,
        value=15.0
    )

    preferred_payment_mode = st.selectbox(
        "Preferred Payment Mode",
        [
            "Debit Card",
            "UPI",
            "CC",
            "Cash on Delivery",
            "E wallet",
            "COD",
            "Credit Card"
        ]
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

with col2:
    hour_spend_on_app = st.number_input(
        "Hours Spent on App",
        min_value=0.0,
        max_value=10.0,
        value=3.0
    )

    number_of_device_registered = st.number_input(
        "Number of Devices Registered",
        min_value=1,
        max_value=10,
        value=3
    )

    prefered_order_cat = st.selectbox(
        "Preferred Order Category",
        [
            "Laptop & Accessory",
            "Mobile Phone",
            "Fashion",
            "Grocery",
            "Others"
        ]
    )

    satisfaction_score = st.slider(
        "Satisfaction Score",
        min_value=1,
        max_value=5,
        value=3
    )

    marital_status = st.selectbox(
        "Marital Status",
        [
            "Single",
            "Married",
            "Divorced"
        ]
    )

    number_of_address = st.number_input(
        "Number of Addresses",
        min_value=1,
        max_value=20,
        value=2
    )

with col3:
    complain = st.selectbox(
        "Has Customer Complained?",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    order_amount_hike = st.number_input(
        "Order Amount Hike from Last Year (%)",
        min_value=0.0,
        max_value=100.0,
        value=15.0
    )

    coupon_used = st.number_input(
        "Coupons Used",
        min_value=0,
        max_value=50,
        value=1
    )

    order_count = st.number_input(
        "Order Count",
        min_value=0.0,
        max_value=100.0,
        value=5.0
    )

    day_since_last_order = st.number_input(
        "Days Since Last Order",
        min_value=0.0,
        max_value=100.0,
        value=5.0
    )

    cashback_amount = st.number_input(
        "Cashback Amount",
        min_value=0.0,
        max_value=1000.0,
        value=150.0
    )


# -----------------------------
# Prediction
# -----------------------------

st.divider()

if st.button(
    "🔮 Predict Customer Churn",
    type="primary",
    use_container_width=True
):

    input_data = pd.DataFrame({
        "Tenure": [tenure],
        "PreferredLoginDevice": [preferred_login_device],
        "CityTier": [city_tier],
        "WarehouseToHome": [warehouse_to_home],
        "PreferredPaymentMode": [preferred_payment_mode],
        "Gender": [gender],
        "HourSpendOnApp": [hour_spend_on_app],
        "NumberOfDeviceRegistered": [number_of_device_registered],
        "PreferedOrderCat": [prefered_order_cat],
        "SatisfactionScore": [satisfaction_score],
        "MaritalStatus": [marital_status],
        "NumberOfAddress": [number_of_address],
        "Complain": [complain],
        "OrderAmountHikeFromlastYear": [order_amount_hike],
        "CouponUsed": [coupon_used],
        "OrderCount": [order_count],
        "DaySinceLastOrder": [day_since_last_order],
        "CashbackAmount": [cashback_amount]
    })

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Customer is likely to churn")
    else:
        st.success("✅ Customer is unlikely to churn")

    st.metric(
        "Churn Probability",
        f"{probability * 100:.2f}%"
    )

st.divider()

st.header("📊 Model Performance")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Accuracy", "98.22%")
col2.metric("Precision", "93.37%")
col3.metric("Recall", "96.32%")
col4.metric("F1-Score", "94.82%")
col5.metric("ROC-AUC", "99.84%")

st.caption(
    "Performance metrics are calculated on the held-out test dataset. "
    "The final prediction model is Random Forest."
)
