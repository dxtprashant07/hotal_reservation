import pandas as pd
from utils.logger import get_logger

logger = get_logger("Preprocess")

def preprocess_data(df):
    """
    Basic preprocessing:
    - Handle missing children count
    - Remove duplicates
    - Fix data types
    - Normalize status (already numeric)
    """
    logger.info("Starting preprocessing...")

    # Fix children missing values
    df['no_of_children'] = df['no_of_children'].fillna(0)

    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)

    logger.info(f"Removed {before - after} duplicate rows")

    # Fix arrival fields if needed
    df['arrival_year'] = df['arrival_year'].astype(int)
    df['arrival_month'] = df['arrival_month'].astype(int)
    df['arrival_date'] = df['arrival_date'].astype(int)

    logger.info("Preprocessing completed")
    return df
