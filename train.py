import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from utils.logger import get_logger
import joblib
import utils.config as cfg

logger = get_logger("Train")


def train_model(df, pipeline):

    # ---------------------------
    # 1. Split data
    # ---------------------------
    logger.info("Splitting data...")

    y = df['booking_status']
    X = df.drop(columns=['booking_status'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ---------------------------
    # 2. Train model
    # ---------------------------
    logger.info("Training model...")
    pipeline.fit(X_train, y_train)

    # ---------------------------
    # 3. Evaluate
    # ---------------------------
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    f1  = f1_score(y_test, y_pred)

    logger.info(f"Accuracy: {acc:.4f}")
    logger.info(f"ROC-AUC: {roc:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")

    # ---------------------------
    # 4. Save model
    # ---------------------------
    logger.info("Saving model...")

    # Ensure model directory exists
    os.makedirs(os.path.dirname(cfg.MODEL_OUTPUT), exist_ok=True)

    # Save to a file instead of folder
    joblib.dump(pipeline, cfg.MODEL_OUTPUT)

    logger.info(f"Model saved to {cfg.MODEL_OUTPUT}")

    return pipeline, X_test, y_test
