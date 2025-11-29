from utils.logger import get_logger
from data_loader import load_data
from preprocess import preprocess_data
from outlier_treatment import apply_outlier_treatment
from model_setup import build_model
from train import train_model
from evaluate import evaluate_model


def main():

    logger = get_logger("Main")
    logger.info("===== Starting ML Pipeline =====")

    # ----------------------------------------------------
    # 1. LOAD DATA
    # ----------------------------------------------------
    df = load_data()

    # ----------------------------------------------------
    # 2. PREPROCESS (cleaning, type conversion, duplicates)
    # ----------------------------------------------------
    df = preprocess_data(df)

    # ----------------------------------------------------
    # 3. OUTLIER TREATMENT
    # (Feature engineering will be done inside FeatureEngineer class)
    # ----------------------------------------------------
    df = apply_outlier_treatment(df)

    # ----------------------------------------------------
    # 4. BUILD MODEL PIPELINE
    # (includes FeatureEngineer + Preprocessing + XGBoost)
    # ----------------------------------------------------
    X = df.drop(columns=['booking_status'])
    pipeline = build_model(X)

    # ----------------------------------------------------
    # 5. TRAIN MODEL
    # (splits data, fits model, saves model)
    # ----------------------------------------------------
    model, X_test, y_test = train_model(df, pipeline)

    # ----------------------------------------------------
    # 6. FINAL EVALUATION
    # ----------------------------------------------------
    evaluate_model(model, X_test, y_test)

    logger.info("===== Pipeline Completed Successfully =====")


if __name__ == "__main__":
    main()
