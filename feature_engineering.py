import pandas as pd
from utils.logger import get_logger

logger = get_logger("FeatureEngineering")

def add_features(df):
    logger.info("Adding engineered features...")

    df = df.copy()

    df['total_nights'] = df['no_of_weekend_nights'] + df['no_of_week_nights']
    df['total_guests'] = df['no_of_adults'] + df['no_of_children']
    df['total_guests'].replace(0, 1, inplace=True)

    df['lead_time_cat'] = pd.cut(
        df['lead_time'],
        bins=[-1, 7, 30, 90, 365, df['lead_time'].max()+1],
        labels=['<1w', '1w-1m', '1m-3m', '3m-1y', '>1y']
    )

    df['adr_per_person'] = df['avg_price_per_room'] / df['total_guests']
    df['is_weekend_booking'] = (df['no_of_weekend_nights'] > 0).astype(int)

    season_map = {
        12:'Winter',1:'Winter',2:'Winter',
        3:'Spring',4:'Spring',5:'Spring',
        6:'Summer',7:'Summer',8:'Summer',
        9:'Fall',10:'Fall',11:'Fall'
    }
    df['arrival_season'] = df['arrival_month'].map(season_map)

    df['is_repeated_guest'] = (df['repeated_guest'] > 0).astype(int)

    logger.info("Feature engineering completed")

    return df
