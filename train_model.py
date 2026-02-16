"""
Train life expectancy prediction model for the Flask app (from task.ipynb).
Requires data.csv in this folder. Run once, then start the app.
"""
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

TARGET = "Life expectancy"


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "data.csv")

    print("Loading data...")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    if TARGET not in df.columns:
        target_col = [c for c in df.columns if "Life" in c and "expectancy" in c.lower()]
        if target_col:
            df = df.rename(columns={target_col[0]: TARGET})

    df = df.dropna(subset=[TARGET])
    for col in df.columns:
        if df[col].dtype in ["float64", "int64"] and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
        elif df[col].dtype == "object" and df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode().iloc[0] if len(df[col].mode()) else "")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    feature_columns = list(X.columns)

    numeric_cols = X.select_dtypes(include=["float64", "int64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ]
    )
    pipeline = Pipeline(
        steps=[
            ("preproc", preprocessor),
            ("reg", LinearRegression()),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Training...")
    pipeline.fit(X_train, y_train)

    pipe_path = os.path.join(base_dir, "pipeline.pkl")
    cols_path = os.path.join(base_dir, "feature_columns.pkl")
    joblib.dump(pipeline, pipe_path)
    joblib.dump(feature_columns, cols_path)
    print(f"Pipeline saved to {pipe_path}")
    print(f"Feature columns ({len(feature_columns)}) saved to {cols_path}")


if __name__ == "__main__":
    main()
