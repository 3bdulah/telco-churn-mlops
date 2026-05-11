"""
Data loading, cleaning, encoding, and splitting for Telco Customer Churn dataset.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")

BINARY_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn",
    "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

CATEGORICAL_COLS = ["InternetService", "Contract", "PaymentMethod"]

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def load_and_preprocess(
    data_path: str = DATA_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
    use_smote: bool = False,
):
    """Load the Telco churn CSV and return train/test arrays ready for modelling.

    Args:
        use_smote: If True, apply SMOTE oversampling to the training set only
                   to address the 73/27 class imbalance.
    """
    df = pd.read_csv(data_path)

    df.drop(columns=["customerID"], inplace=True)

    # TotalCharges can be blank strings — coerce and fill
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    # Binary Yes/No → 1/0; gender Male→1, Female→0
    binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0,
                  "No phone service": 0, "No internet service": 0}
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].map(binary_map).fillna(df[col])

    # One-hot encode multi-class categoricals
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=False)

    # Scale numeric features
    scaler = StandardScaler()
    df[NUMERIC_COLS] = scaler.fit_transform(df[NUMERIC_COLS])

    y = df["Churn"].astype(int).values
    X = df.drop(columns=["Churn"])
    feature_names = list(X.columns)
    X = X.values.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if use_smote:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=random_state)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    return X_train, X_test, y_train, y_test, feature_names, scaler


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feature_names, _ = load_and_preprocess(use_smote=True)
    print(f"Train size : {X_train.shape}")
    print(f"Test size  : {X_test.shape}")
    print(f"Features   : {len(feature_names)}")
    print(f"Churn rate (train after SMOTE): {y_train.mean():.2%}")
