import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import shap

from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# BASE DIRECTORY (makes file paths work regardless of where
# the app is launched from — required for cloud deployment)
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# FEATURE ENGINEERING (mirrors Step 6 from the notebook)
# ============================================================

def engineer_features(input_df):
    """
    Recreates the exact engineered features from Step 6
    so raw form input matches what the pipeline was trained on.
    """
    df = input_df.copy()

    # TenureGroup
    def tenure_group(tenure):
        if tenure <= 12:
            return "0-1 year"
        elif tenure <= 24:
            return "1-2 years"
        elif tenure <= 48:
            return "2-4 years"
        else:
            return "4+ years"

    df["TenureGroup"] = df["tenure"].apply(tenure_group)

    # NumServices
    service_cols = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]
    df["NumServices"] = (df[service_cols] == "Yes").sum(axis=1)

    # HasInternetAndPhone
    df["HasInternetAndPhone"] = (
        (df["InternetService"] != "No") & (df["PhoneService"] == "Yes")
    ).astype(int)

    # AvgMonthlySpend
    df["AvgMonthlySpend"] = df["TotalCharges"] / df["tenure"].replace(0, 1)

    return df


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    # Load preprocessing pipeline
    preprocessor = joblib.load(
        os.path.join(BASE_DIR, "..", "models", "preprocessor.joblib")
    )

    # Load trained XGBoost model
    xgb_model = XGBClassifier()

    xgb_model.load_model(
        os.path.join(BASE_DIR, "..", "models", "xgb_model.json")
    )

    # Combine preprocessing + model
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", xgb_model)
        ]
    )

    return pipeline


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        os.path.join(BASE_DIR, "..", "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    )

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Customers with zero tenure
    df.loc[
        df["tenure"] == 0,
        "TotalCharges"
    ] = df.loc[
        df["tenure"] == 0,
        "TotalCharges"
    ].fillna(0)

    return df


# ============================================================
# LOAD MODEL AND DATA
# ============================================================

pipeline = load_model()
raw_df = load_data()


# ============================================================
# TITLE
# ============================================================

st.title("📊 Customer Churn Prediction & Analysis")

st.markdown(
    """
    ### AI-powered customer churn analysis system

    This application uses **Machine Learning and XGBoost**
    to predict whether a customer is likely to churn.
    """
)


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Navigate",
    [
        "Dashboard",
        "Churn Prediction"
    ]
)


# ============================================================
# DASHBOARD PAGE
# ============================================================

if page == "Dashboard":

    st.header("📈 Customer Churn Dashboard")

    # --------------------------------------------------------
    # KPI CALCULATIONS
    # --------------------------------------------------------

    total_customers = len(raw_df)

    churned_customers = (
        raw_df["Churn"] == "Yes"
    ).sum()

    retained_customers = (
        raw_df["Churn"] == "No"
    ).sum()

    churn_rate = (
        churned_customers / total_customers
    ) * 100

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "👥 Total Customers",
        f"{total_customers:,}"
    )

    col2.metric(
        "🔴 Churned Customers",
        f"{churned_customers:,}"
    )

    col3.metric(
        "🟢 Retained Customers",
        f"{retained_customers:,}"
    )

    col4.metric(
        "📉 Churn Rate",
        f"{churn_rate:.2f}%"
    )

    st.divider()

    # --------------------------------------------------------
    # CHURN DISTRIBUTION
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Customer Churn Distribution")

        churn_counts = (
            raw_df["Churn"]
            .value_counts()
            .reset_index()
        )

        churn_counts.columns = [
            "Churn",
            "Customers"
        ]

        fig = px.pie(
            churn_counts,
            names="Churn",
            values="Customers",
            hole=0.4,
            title="Churn vs Retention"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # CONTRACT TYPE
    # --------------------------------------------------------

    with col2:

        st.subheader("Churn by Contract")

        contract_churn = (
            raw_df.groupby(
                ["Contract", "Churn"]
            )
            .size()
            .reset_index(
                name="Customers"
            )
        )

        fig = px.bar(
            contract_churn,
            x="Contract",
            y="Customers",
            color="Churn",
            barmode="group",
            title="Churn by Contract Type"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # TENURE ANALYSIS
    # --------------------------------------------------------

    st.subheader("Customer Tenure Distribution")

    fig = px.histogram(
        raw_df,
        x="tenure",
        color="Churn",
        nbins=30,
        title="Tenure vs Churn"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # MONTHLY CHARGES
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Monthly Charges")

        fig = px.box(
            raw_df,
            x="Churn",
            y="MonthlyCharges",
            color="Churn",
            title="Monthly Charges vs Churn"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        st.subheader("Total Charges")

        fig = px.box(
            raw_df,
            x="Churn",
            y="TotalCharges",
            color="Churn",
            title="Total Charges vs Churn"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # SENIOR CITIZEN ANALYSIS
    # --------------------------------------------------------

    st.subheader("Churn by Senior Citizen")

    senior_df = (
        raw_df.groupby(
            ["SeniorCitizen", "Churn"]
        )
        .size()
        .reset_index(
            name="Customers"
        )
    )

    senior_df["SeniorCitizen"] = (
        senior_df["SeniorCitizen"]
        .map({
            0: "Non-Senior",
            1: "Senior"
        })
    )

    fig = px.bar(
        senior_df,
        x="SeniorCitizen",
        y="Customers",
        color="Churn",
        barmode="group",
        title="Churn Distribution by Senior Citizen"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# CHURN PREDICTION PAGE
# ============================================================

elif page == "Churn Prediction":

    st.header("🤖 Customer Churn Prediction")

    st.markdown(
        """
        Enter the customer's information below and the
        AI model will estimate the probability of churn.
        """
    )

    # --------------------------------------------------------
    # CUSTOMER INFORMATION
    # --------------------------------------------------------

    st.subheader("👤 Customer Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1]
        )

        partner = st.selectbox(
            "Partner",
            ["Yes", "No"]
        )

    with col2:

        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"]
        )

        tenure = st.number_input(
            "Tenure (months)",
            min_value=0,
            max_value=100,
            value=12
        )

        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

    with col3:

        multiple_lines = st.selectbox(
            "Multiple Lines",
            [
                "Yes",
                "No",
                "No phone service"
            ]
        )

        internet_service = st.selectbox(
            "Internet Service",
            [
                "DSL",
                "Fiber optic",
                "No"
            ]
        )

        online_security = st.selectbox(
            "Online Security",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

    # --------------------------------------------------------
    # SERVICES
    # --------------------------------------------------------

    st.subheader("🌐 Internet & Additional Services")

    col1, col2, col3 = st.columns(3)

    with col1:

        online_backup = st.selectbox(
            "Online Backup",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        device_protection = st.selectbox(
            "Device Protection",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        tech_support = st.selectbox(
            "Tech Support",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

    with col2:

        streaming_tv = st.selectbox(
            "Streaming TV",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

    with col3:

        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    # --------------------------------------------------------
    # BILLING
    # --------------------------------------------------------

    st.subheader("💰 Billing Information")

    col1, col2 = st.columns(2)

    with col1:

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=70.0,
            step=1.0
        )

    with col2:

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=monthly_charges * tenure,
            step=10.0
        )

    # --------------------------------------------------------
    # PREDICTION BUTTON
    # --------------------------------------------------------

    st.divider()

    predict_button = st.button(
        "🔮 Predict Customer Churn",
        type="primary",
        use_container_width=True
    )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if predict_button:

        # Create customer dataframe (raw form input, same shape as original CSV columns)
        customer_data = pd.DataFrame({

            "gender": [gender],

            "SeniorCitizen": [senior_citizen],

            "Partner": [partner],

            "Dependents": [dependents],

            "tenure": [tenure],

            "PhoneService": [phone_service],

            "MultipleLines": [multiple_lines],

            "InternetService": [internet_service],

            "OnlineSecurity": [online_security],

            "OnlineBackup": [online_backup],

            "DeviceProtection": [device_protection],

            "TechSupport": [tech_support],

            "StreamingTV": [streaming_tv],

            "StreamingMovies": [streaming_movies],

            "Contract": [contract],

            "PaperlessBilling": [paperless_billing],

            "PaymentMethod": [payment_method],

            "MonthlyCharges": [monthly_charges],

            "TotalCharges": [total_charges]
        })

        # ----------------------------------------------------
        # FEATURE ENGINEERING (must match training exactly)
        # ----------------------------------------------------

        customer_data = engineer_features(customer_data)

        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        try:

            prediction = pipeline.predict(
                customer_data
            )

            probability = pipeline.predict_proba(
                customer_data
            )[0][1]

            probability_percentage = (
                probability * 100
            )

            # ------------------------------------------------
            # DISPLAY RESULT
            # ------------------------------------------------

            st.subheader("📊 Prediction Result")

            col1, col2 = st.columns(2)

            with col1:

                if prediction[0] == 1:

                    st.error(
                        "⚠️ Customer is likely to CHURN"
                    )

                else:

                    st.success(
                        "✅ Customer is likely to STAY"
                    )

            with col2:

                st.metric(
                    "Churn Probability",
                    f"{probability_percentage:.2f}%"
                )

            # ------------------------------------------------
            # PROBABILITY BAR
            # ------------------------------------------------

            st.progress(
                float(probability)
            )

            # ------------------------------------------------
            # RISK LEVEL
            # ------------------------------------------------

            if probability >= 0.75:

                st.error(
                    "🔴 HIGH RISK — Immediate retention action recommended."
                )

            elif probability >= 0.50:

                st.warning(
                    "🟠 MEDIUM RISK — Customer should be monitored."
                )

            else:

                st.success(
                    "🟢 LOW RISK — Customer is unlikely to churn."
                )

            # ------------------------------------------------
            # RECOMMENDATIONS
            # ------------------------------------------------

            st.subheader(
                "💡 Recommended Actions"
            )

            if probability >= 0.75:

                st.write(
                    """
                    - Contact the customer immediately.
                    - Offer a personalized discount.
                    - Provide a contract upgrade.
                    - Investigate service issues.
                    - Offer loyalty benefits.
                    """
                )

            elif probability >= 0.50:

                st.write(
                    """
                    - Monitor the customer.
                    - Offer additional support.
                    - Consider a loyalty offer.
                    - Review monthly charges.
                    """
                )

            else:

                st.write(
                    """
                    - Continue normal customer engagement.
                    - Maintain service quality.
                    - Consider loyalty rewards.
                    """
                )

        except Exception as e:

            st.error(
                "Prediction failed."
            )

            st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.info(
    """
    **Customer Churn Prediction System**

    Machine Learning Model:
    XGBoost

    Dataset:
    Telco Customer Churn

    Built with:
    Python + Streamlit
    """
)