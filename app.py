# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import traceback

st.set_page_config(page_title="Hotel Cancellation Predictor", layout="wide")

DEFAULT_MODEL_PATH = "model/final_model.joblib"

st.title("🏨 Hotel Booking Cancellation Predictor")

# -------------------------------------------------------------
# Utility: Auto-detect preprocessor + classifier in ANY pipeline
# -------------------------------------------------------------
def find_step(pipeline, step_type="preprocessor"):
    """
    Automatically locate a pipeline step by type.
    step_type = 'preprocessor' or 'classifier'
    """
    for name, step in pipeline.named_steps.items():
        if step_type == "preprocessor":
            # ColumnTransformer or similar
            from sklearn.compose import ColumnTransformer
            if isinstance(step, ColumnTransformer):
                return name, step

        if step_type == "classifier":
            # XGBoost, RandomForest, LogisticRegression, etc.
            from sklearn.base import ClassifierMixin
            if isinstance(step, ClassifierMixin):
                return name, step

    return None, None  # not found


# -------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(path=DEFAULT_MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at: {path}")
    return joblib.load(path)


def predict_df(pipeline, df):
    proba = pipeline.predict_proba(df)[:, 1]
    preds = (proba >= 0.5).astype(int)

    out = df.copy().reset_index(drop=True)
    out["predicted_canceled"] = preds
    out["canceled_proba"] = proba
    return out


def get_feature_names(preprocessor):
    """Extract readable feature names."""
    try:
        feature_names = []
        for name, trans, cols in preprocessor.transformers_:
            if name == "num":
                feature_names.extend(list(cols))
            elif name == "low_cat":
                ohe = trans.named_steps.get("ohe")
                if ohe is not None:
                    try:
                        feature_names.extend(ohe.get_feature_names_out(cols))
                    except:
                        feature_names.extend(list(cols))
            elif name == "high_cat":
                feature_names.extend(list(cols))
        return feature_names
    except:
        return None  # fallback


# -------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------
st.sidebar.header("Model / Data")
model_path = st.sidebar.text_input("Model path", value=DEFAULT_MODEL_PATH)
load_button = st.sidebar.button("Load model")

model = None
if load_button:
    try:
        model = load_model(model_path)
        st.sidebar.success("Model loaded successfully.")
    except Exception as e:
        st.sidebar.error(f"Failed to load model: {e}")
        st.sidebar.text(traceback.format_exc())

if model is None:
    try:
        model = load_model(model_path)
        st.sidebar.success("Model auto-loaded.")
    except:
        st.sidebar.warning("Model not loaded. Enter correct model path.")


# -------------------------------------------------------------
# CSV Upload Prediction
# -------------------------------------------------------------
st.header("1. Upload CSV for Batch Prediction")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df_input = pd.read_csv(uploaded_file)
    st.dataframe(df_input.head())

    if st.button("Run predictions"):
        try:
            preds_df = predict_df(model, df_input)
            st.dataframe(preds_df.head())
            st.download_button("Download Predictions", preds_df.to_csv(index=False), "preds.csv")
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.text(traceback.format_exc())


# -------------------------------------------------------------
# Single Row Prediction Form
# -------------------------------------------------------------
st.header("2. Predict Single Booking")

with st.form("single_form"):
    booking_id = st.text_input("Booking_ID", "INN00000")
    no_of_adults = st.number_input("no_of_adults", 0, 10, 2)
    no_of_children = st.number_input("no_of_children", 0, 10, 0)
    no_of_weekend_nights = st.number_input("no_of_weekend_nights", 0, 10, 1)
    no_of_week_nights = st.number_input("no_of_week_nights", 0, 20, 2)
    type_of_meal_plan = st.selectbox("type_of_meal_plan", ["Not Selected","Meal Plan 1","Meal Plan 2"])
    required_car_parking_space = st.number_input("required_car_parking_space", 0, 10, 0)
    room_type_reserved = st.selectbox("room_type_reserved", ["Room_Type 1","Room_Type 2","Room_Type 3"])
    lead_time = st.number_input("lead_time", 0, 365, 20)
    arrival_year = st.number_input("arrival_year", 2000, 2050, 2024)
    arrival_month = st.number_input("arrival_month", 1, 12, 6)
    arrival_date = st.number_input("arrival_date", 1, 31, 10)
    market_segment_type = st.selectbox("market_segment_type", ["Offline","Online","Corporate"])
    repeated_guest = st.number_input("repeated_guest", 0, 10, 0)
    no_of_previous_cancellations = st.number_input("no_of_previous_cancellations", 0, 10, 0)
    no_of_previous_bookings_not_canceled = st.number_input("no_of_previous_bookings_not_canceled", 0, 20, 0)
    avg_price_per_room = st.number_input("avg_price_per_room", 0.0, 2000.0, 100.0)
    no_of_special_requests = st.number_input("no_of_special_requests", 0, 5, 0)

    submit = st.form_submit_button("Predict")

    if submit:
        row = pd.DataFrame([{
            "Booking_ID": booking_id,
            "no_of_adults": no_of_adults,
            "no_of_children": no_of_children,
            "no_of_weekend_nights": no_of_weekend_nights,
            "no_of_week_nights": no_of_week_nights,
            "type_of_meal_plan": type_of_meal_plan,
            "required_car_parking_space": required_car_parking_space,
            "room_type_reserved": room_type_reserved,
            "lead_time": lead_time,
            "arrival_year": arrival_year,
            "arrival_month": arrival_month,
            "arrival_date": arrival_date,
            "market_segment_type": market_segment_type,
            "repeated_guest": repeated_guest,
            "no_of_previous_cancellations": no_of_previous_cancellations,
            "no_of_previous_bookings_not_canceled": no_of_previous_bookings_not_canceled,
            "avg_price_per_room": avg_price_per_room,
            "no_of_special_requests": no_of_special_requests
        }])

        try:
            out = predict_df(model, row)
            st.metric("Prediction (0=no,1=yes)", int(out["predicted_canceled"].iloc[0]))
            st.metric("Cancellation probability", float(out["canceled_proba"].iloc[0]))
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.text(traceback.format_exc())


# -------------------------------------------------------------
# Feature Importances
# -------------------------------------------------------------
st.header("Model Insights")

if st.button("Show Feature Importances"):
    name_p, preproc = find_step(model, "preprocessor")
    name_c, clf = find_step(model, "classifier")

    if clf is None:
        st.error("Could not locate classifier in pipeline.")
    else:
        try:
            feat_names = get_feature_names(preproc) if preproc else None
            importances = clf.feature_importances_
            df_imp = pd.DataFrame({
                "feature": feat_names if feat_names else range(len(importances)),
                "importance": importances
            }).sort_values("importance", ascending=False)

            st.dataframe(df_imp.head(30))
        except:
            st.error("Classifier doesn't support feature_importances_.")


# -------------------------------------------------------------
# SHAP
# -------------------------------------------------------------
st.header("SHAP Explanations")

if st.button("Compute SHAP (sample)"):
    try:
        import shap

        name_p, preproc = find_step(model, "preprocessor")
        name_c, clf = find_step(model, "classifier")

        if preproc is None or clf is None:
            st.error("Could not detect preprocessor or classifier in pipeline.")
        else:
            st.info("Upload a CSV first to sample from it.")
            if "df_input" not in locals():
                st.warning("No CSV uploaded. Upload CSV to compute SHAP.")
            else:
                sample = df_input.sample(min(200, len(df_input)), random_state=42)
                X_trans = preproc.transform(sample)

                explainer = shap.Explainer(clf)
                shap_values = explainer(X_trans)


    except Exception as e:
        st.error("SHAP failed.")
        st.text(traceback.format_exc())
