from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb


def build_preprocessor(X):
    num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object','category']).columns.tolist()

    high_card = [c for c in cat_cols if X[c].nunique() > 20]
    low_card = [c for c in cat_cols if c not in high_card]

    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    low_cat = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='MISSING')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    high_cat = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='MISSING')),
        ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipe, num_cols),
            ('low_cat', low_cat, low_card),
            ('high_cat', high_cat, high_card)
        ],
        remainder='drop'
    )

    return preprocessor



def build_model(preprocessor):
    # 🔥 Hyper-Tuned XGBoost Model
    xgb_best = xgb.XGBClassifier(
        eval_metric='logloss',
        learning_rate=0.1,
        n_estimators=300,
        max_depth=9,
        subsample=0.6,
        colsample_bytree=1.0,
        gamma=1,
        random_state=42
    )

    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('clf', xgb_best)
    ])

    return pipeline
