# import numpy as np
# from utils.logger import get_logger

# logger = get_logger("OutlierTreatment")

# def treat_outliers(df):
#     logger.info("Applying outlier treatment...")

#     df = df.copy()

#     lt_99 = df['lead_time'].quantile(0.99)
#     adr_99 = df['avg_price_per_room'].quantile(0.99)

#     df['lead_time_capped'] = np.where(df['lead_time'] > lt_99, lt_99, df['lead_time'])
#     df['avg_price_capped'] = np.where(df['avg_price_per_room'] > adr_99, adr_99, df['avg_price_per_room'])
#     df['adr_per_person_capped'] = np.where(df['adr_per_person'] > adr_99, adr_99, df['adr_per_person'])

#     df['log_avg_price'] = np.log1p(df['avg_price_capped'])
#     df['log_adr_per_person'] = np.log1p(df['adr_per_person_capped'])

#     logger.info("Outlier treatment completed")

#     return df


from utils.logger import get_logger

logger = get_logger("OutlierTreatment")

def apply_outlier_treatment(df):
    """
    Outlier treatment is now handled inside FeatureEngineer.
    This function exists only to keep main.py compatibility.
    It returns the dataframe unchanged.
    """
    logger.info("Outlier treatment skipped (handled in FeatureEngineer)")
    return df
