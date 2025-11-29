from data_loader import load_data
from preprocess import preprocess
from feature_engineering import add_features
from outlier_treatment import treat_outliers
from model_setup import build_preprocessor, build_model
from train import train_model
from evaluate import evaluate_model
from utils.logger import get_logger

logger = get_logger("Main")

def main():
    logger.info("===== Starting ML Pipeline =====")

    df = load_data()
    df = preprocess(df)
    df = add_features(df)
    df = treat_outliers(df)

    y = df['booking_status']
    X = df.drop(columns=['booking_status'])

    preprocessor = build_preprocessor(X)
    pipeline = build_model(preprocessor)

    model, X_test, y_test = train_model(df, pipeline)
    evaluate_model(model, X_test, y_test)

    logger.info("===== Pipeline Completed Successfully =====")

if __name__ == "__main__":
    main()
