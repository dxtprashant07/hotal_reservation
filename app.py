# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import io
import traceback

st.set_page_config(page_title="Hotel Cancellation Predictor", layout="wide")

DEFAULT_MODEL_PATH = "model/final_model.joblib"

st.title("🏨 Hotel Booking Cancellation Predictor")
st.markdown(
    """
    Upload a CSV of raw booking rows (same columns as training) or fill the form to predict whether a booking will be canceled.
    The app expects the saved pipeline to be a scikit-learn / imblearn pipeline that accepts raw DataFrame rows.
    """
)

@st.cache_resource(show_spinner=False)
def load_model(path=DEFAULT_MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at: {path}")
    model = joblib.load(path)
    return model

def predict_df(pipeline, df):
    """Return dataframe with predictions and probabilities appended."""
    proba = pipeline.predict_proba(df)[:, 1]
    preds = (proba >= 0.5).astype(int)
    out = df.copy().reset_index(drop=True)
    out["predicted_canceled"] = preds
    out["canceled_proba"] = proba
    return out

def get_feature_names_from_pipeline(pipeline):
    """Attempt to extract human readable feature names after preprocessing."""
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
    except Exception:
        return None

    feature_names = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == "num":
            feature_names.extend(list(cols))
        elif name == "low_cat":
            # OneHotEncoder present
            ohe = transformer.named_steps.get("ohe", None)
            if ohe is not None:
                try:
                    ohe_names = ohe.get_feature_names_out(cols)
                    feature_names.extend(list(ohe_names))
                except Exception:
                    feature_names.extend(list(cols))
            else:
                feature_names.extend(list(cols))
        elif name == "high_cat":
            feature_names.extend(list(cols))
    return feature_names

# Sidebar: model selection and upload
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

# Try autoload if not loaded yet (first load)
if model is None:
    try:
        model = load_model(model_path)
        st.sidebar.success("Model auto-loaded.")
    except Exception as e:
        st.sidebar.warning("Model not auto-loaded. Click 'Load model' in the sidebar or fix path.")
        # show minimal info
        st.sidebar.info(f"Default path: {DEFAULT_MODEL_PATH}")

# Main: Upload CSV or single-row input
st.header("1. Upload CSV (batch predictions)")
uploaded_file = st.file_uploader("Upload CSV file with booking rows (same columns as training)", type=["csv"])

if uploaded_file is not None:
    try:
        df_input = pd.read_csv(uploaded_file)
        st.subheader("Preview uploaded data (first 5 rows)")
        st.dataframe(df_input.head())

        if st.button("Run predictions on uploaded CSV"):
            try:
                if model is None:
                    st.error("Model not loaded. Please provide a valid model path in the sidebar.")
                else:
                    preds_df = predict_df(model, df_input)
                    st.success("Predictions completed.")
                    st.dataframe(preds_df.head(50))

                    # Download link
                    csv = preds_df.to_csv(index=False).encode("utf-8")
                    st.download_button("Download predictions CSV", data=csv, file_name="predictions.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.text(traceback.format_exc())

    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.text(traceback.format_exc())

st.markdown("---")
st.header("2. Predict single booking (manual entry)")

# We'll create inputs for the most important fields — if you have more columns, user can upload CSV instead
with st.form("single_row_form", clear_on_submit=False):
    st.write("Fill the booking fields (use same column names as training). If a column is missing, fill with a reasonable default.")
    # Note: customize these inputs to match your dataset schema
    booking_id = st.text_input("Booking_ID", value="INN00000")
    no_of_adults = st.number_input("no_of_adults", min_value=0, value=2)
    no_of_children = st.number_input("no_of_children", min_value=0, value=0)
    no_of_weekend_nights = st.number_input("no_of_weekend_nights", min_value=0, value=1)
    no_of_week_nights = st.number_input("no_of_week_nights", min_value=0, value=2)
    type_of_meal_plan = st.selectbox("type_of_meal_plan", options=["Not Selected","Meal Plan 1","Meal Plan 2","Meal Plan 3"])
    required_car_parking_space = st.number_input("required_car_parking_space", min_value=0, value=0)
    room_type_reserved = st.selectbox("room_type_reserved", options=["Room_Type 1","Room_Type 2","Room_Type 3","Room_Type 4","Room_Type 5"])
    lead_time = st.number_input("lead_time", min_value=0, value=30)
    arrival_year = st.number_input("arrival_year", min_value=2000, max_value=2100, value=2025)
    arrival_month = st.number_input("arrival_month", min_value=1, max_value=12, value=6)
    arrival_date = st.number_input("arrival_date", min_value=1, max_value=31, value=15)
    market_segment_type = st.selectbox("market_segment_type", options=["Offline","Online","Corporate","Complementary"])
    repeated_guest = st.number_input("repeated_guest", min_value=0, value=0)
    no_of_previous_cancellations = st.number_input("no_of_previous_cancellations", min_value=0, value=0)
    no_of_previous_bookings_not_canceled = st.number_input("no_of_previous_bookings_not_canceled", min_value=0, value=0)
    avg_price_per_room = st.number_input("avg_price_per_room", min_value=0.0, value=100.0, format="%.2f")
    no_of_special_requests = st.number_input("no_of_special_requests", min_value=0, value=0)

    submitted = st.form_submit_button("Predict single booking")
    if submitted:
        # assemble a dataframe (single row)
        row = {
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
            "no_of_special_requests": no_of_special_requests,
        }
        input_df = pd.DataFrame([row])
        st.write("Input row:")
        st.dataframe(input_df)

        if model is None:
            st.error("Model not loaded. Please supply a valid model path in the sidebar.")
        else:
            try:
                out = predict_df(model, input_df)
                st.metric("Predicted Canceled (0=No,1=Yes)", int(out["predicted_canceled"].iloc[0]))
                st.metric("Cancellation probability", float(out["canceled_proba"].iloc[0]))
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.text(traceback.format_exc())

st.markdown("---")
st.header("Model Insights")

if st.button("Show feature importances (if supported)"):
    if model is None:
        st.error("Model not loaded.")
    else:
        try:
            clf = model.named_steps.get("clf", None)
            preproc = model.named_steps.get("preprocessor", None)
            if clf is None or preproc is None:
                st.error("Pipeline structure unexpected: cannot extract feature importances.")
            else:
                feat_names = get_feature_names_from_pipeline(model)
                if feat_names is None:
                    st.warning("Could not extract feature names from the preprocessor.")
                else:
                    importances = clf.feature_importances_
                    fi = pd.Series(importances, index=feat_names).sort_values(ascending=False).head(30)
                    st.dataframe(fi.to_frame("importance"))
        except Exception as e:
            st.error(f"Feature importance extraction failed: {e}")
            st.text(traceback.format_exc())

# Optional SHAP (if installed)
st.markdown("---")
st.write("Optional: SHAP explanations (requires `shap` installed).")
if st.button("Run SHAP summary (sample)"):
    if model is None:
        st.error("Model not loaded.")
    else:
        try:
            import shap
            st.info("Computing SHAP values on a small sample (may take a while).")
            # prepare a small sample from local dataset if available (not uploaded)
            # If user uploaded a file use it
            if 'df_input' in locals():
                sample = df_input.sample(min(200, len(df_input)), random_state=42)
            else:
                st.warning("No dataset uploaded to sample from. Please upload a CSV first.")
                sample = None

            if sample is not None:
                preproc = model.named_steps['preprocessor']
                clf = model.named_steps['clf']
                X_trans = preproc.transform(sample)
                explainer = shap.Explainer(clf)
                shap_values = explainer(X_trans[:200])
                st.pyplot(shap.plots.beeswarm(shap_values, show=False))
        except Exception as e:
            st.error("SHAP not available or failed.")
            st.text(traceback.format_exc())

st.markdown("---")
st.write("If you want any UI changes (more fields, nicer layout, threshold slider), tell me which.")
