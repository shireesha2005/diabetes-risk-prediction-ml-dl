import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from src.preprocess import load_data, preprocess_data


def train_ml_models():
    # Load data
    df = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    # 🔹 Random Forest (controlled depth)
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    rf_model.fit(X_train, y_train)

    # 🔹 XGBoost (controlled depth)
    xgb_model = XGBClassifier(
        max_depth=3,
        n_estimators=100,
        learning_rate=0.1,
        eval_metric='logloss',
        random_state=42
    )
    xgb_model.fit(X_train, y_train)

    # 🔥 Save models
    joblib.dump(rf_model, "rf_model.pkl")
    joblib.dump(xgb_model, "xgb_model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    print("\n✅ ML Models trained and saved successfully!")


if __name__ == "__main__":
    train_ml_models() 