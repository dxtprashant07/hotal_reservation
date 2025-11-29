import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Reproduces EXACT same engineered features used in training notebook.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self  # nothing to fit

    def transform(self, X):
        df = X.copy()

        # ------------------------------
        # Replace missing values
        # ------------------------------
        df['no_of_children'] = df['no_of_children'].fillna(0)

        # ------------------------------
        # 1. Total nights
        # ------------------------------
        df['total_nights'] = df['no_of_weekend_nights'] + df['no_of_week_nights']

        # ------------------------------
        # 2. Total guests
        # ------------------------------
        df['total_guests'] = df['no_of_adults'] + df['no_of_children']
        df.loc[df['total_guests'] == 0, 'total_guests'] = 1

        # ------------------------------
        # 3. Lead time category
        # ------------------------------
        # FIXED LEAD TIME BINS (always increasing)
        bins = [-1, 7, 30, 90, 365, 2000]   # last bin is VERY large, always > any lead_time
        labels = ['<1w', '1w-1m', '1m-3m', '3m-1y', '>1y']
        df['lead_time_cat'] = pd.cut(df['lead_time'], bins=bins, labels=labels)


        # ------------------------------
        # 4. ADR per person
        # ------------------------------
        df['adr_per_person'] = df['avg_price_per_room'] / df['total_guests']

        # ------------------------------
        # 5. Weekend booking
        # ------------------------------
        df['is_weekend_booking'] = (df['no_of_weekend_nights'] > 0).astype(int)

        # ------------------------------
        # 6. Arrival season
        # ------------------------------
        season_map = {
            12:'Winter',1:'Winter',2:'Winter',
            3:'Spring',4:'Spring',5:'Spring',
            6:'Summer',7:'Summer',8:'Summer',
            9:'Fall',10:'Fall',11:'Fall'
        }
        df['arrival_season'] = df['arrival_month'].map(season_map)

        # ------------------------------
        # 7. Repeated guest binary
        # ------------------------------
        df['is_repeated_guest'] = (df['repeated_guest'] > 0).astype(int)

        # ------------------------------
        # Outlier capping
        # ------------------------------
        lt_99 = df['lead_time'].quantile(0.99)
        adr_99 = df['avg_price_per_room'].quantile(0.99)

        df['lead_time_capped'] = np.where(df['lead_time'] > lt_99, lt_99, df['lead_time'])
        df['avg_price_capped'] = np.where(df['avg_price_per_room'] > adr_99, adr_99, df['avg_price_per_room'])
        df['adr_per_person_capped'] = np.where(df['adr_per_person'] > adr_99, adr_99, df['adr_per_person'])

        # log transforms
        df['log_avg_price'] = np.log1p(df['avg_price_capped'])
        df['log_adr_per_person'] = np.log1p(df['adr_per_person_capped'])

        return df
