import pandas as pd
from utils.logger import get_logger
import utils.config as cfg

logger = get_logger("DataLoader")

def load_data():
    logger.info("Loading dataset...")
    df = pd.read_csv(cfg.DATA_PATH)

    logger.info(f"Dataset loaded with shape: {df.shape}")

    # Convert booking_status to numeric
    df['booking_status'] = df['booking_status'].map({
        'Canceled': 1,
        'Not_Canceled': 0
    })

    logger.info("booking_status column converted to numeric")

    return df
