from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from utils.logger import get_logger

logger = get_logger("Evaluate")

def evaluate_model(model, X_test, y_test):
    logger.info("Evaluating model...")
    
    y_pred = model.predict(X_test)

    print("\n=== Classification Report ===\n")
    print(classification_report(y_test, y_pred))

    disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()
