from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from utils.logger import get_logger
import joblib
import utils.config as cfg

logger = get_logger("Train")

def train_model(df, pipeline):
    logger.info("Splitting data...")

    y = df['booking_status']
    X = df.drop(columns=['booking_status'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    logger.info("Training model...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    logger.info(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    logger.info(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
    logger.info(f"F1 Score: {f1_score(y_test, y_pred):.4f}")

    logger.info("Saving model...")
    joblib.dump(pipeline, cfg.MODEL_OUTPUT)

    return pipeline, X_test, y_test
