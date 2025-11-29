import pandas as pd
from utils.logger import get_logger

logger = get_logger("Preprocess")

def preprocess(df):
    logger.info("Starting preprocessing...")

    df = df.copy()

    # Replace missing values
    df['no_of_children'] = df['no_of_children'].fillna(0)

    # Remove duplicates
    dup_count = df.duplicated().sum()
    df = df.drop_duplicates()

    logger.info(f"Removed {dup_count} duplicate rows")

    # Convert datatypes
    for col in ['arrival_year', 'arrival_month', 'arrival_date']:
        df[col] = df[col].astype(int)

    df.reset_index(drop=True, inplace=True)

    logger.info("Preprocessing completed")

    return df
