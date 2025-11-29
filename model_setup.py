import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from xgboost import XGBClassifier

from feature_transformer import FeatureEngineer


def build_preprocessor(X):
    """
    Builds the preprocessing pipeline:
    - FeatureEngineer
    - Numerical scaling
    - Categorical encoding
    """

    # Identify column types
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Determine low & high cardinality categorical columns
    low_card = [c for c in cat_cols if X[c].nunique() <= 20]
    high_card = [c for c in cat_cols if c not in low_card]

    # --- Pipelines ---
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    low_cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ])

    high_cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_cols),
            ('low_cat', low_cat_pipeline, low_card),
            ('high_cat', high_cat_pipeline, high_card)
        ],
        remainder='drop'
    )

    return preprocessor



def build_model(X):
    """
    Builds full ML pipeline including:
    - Feature Engineering
    - Preprocessing
    - Tuned XGBoost model
    """

    preprocessor = build_preprocessor(X)

    # Your tuned hyperparameters (already proven best)
    tuned_xgb = XGBClassifier(
        n_estimators=300,
        max_depth=9,
        learning_rate=0.1,
        subsample=0.6,
        colsample_bytree=1.0,
        gamma=1,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    # Final full pipeline
    pipeline = Pipeline([
        ('features', FeatureEngineer()),   # NEW: inject all feature engineering
        ('preprocess', preprocessor),
        ('model', tuned_xgb)
    ])

    return pipeline
