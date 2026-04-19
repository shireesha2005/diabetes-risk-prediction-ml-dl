import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from src.preprocess import load_data, preprocess_data


# 🔹 Print metrics + confusion matrix
def print_metrics(y_true, y_pred, name):
    print(f"\n--- {name} Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.2f}")
    print(f"Precision: {precision_score(y_true, y_pred):.2f}")
    print(f"Recall: {recall_score(y_true, y_pred):.2f}")
    print(f"F1 Score: {f1_score(y_true, y_pred):.2f}")

    cm = confusion_matrix(y_true, y_pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


# 🔹 User Input Prediction
def predict_user_input(model, scaler):
    print("\n--- Enter Patient Details ---")

    pregnancies = int(input("Pregnancies: "))
    glucose = float(input("Glucose: "))
    bp = float(input("Blood Pressure: "))
    skin = float(input("Skin Thickness: "))
    insulin = float(input("Insulin: "))
    bmi = float(input("BMI: "))
    dpf = float(input("Diabetes Pedigree Function: "))
    age = int(input("Age: "))

    user_data = [[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]]

    # Scale input
    user_scaled = scaler.transform(user_data)

    prediction = model.predict(user_scaled)[0]

    print("\nPrediction (0 = Not Diabetic, 1 = Diabetic):", int(prediction))


# 🔹 Main Evaluation Function
def evaluate_ml_models():
    # Load data
    df = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    # Load models
    rf_model = joblib.load("rf_model.pkl")
    xgb_model = joblib.load("xgb_model.pkl")
    scaler = joblib.load("scaler.pkl")

    # Predictions
    rf_pred = rf_model.predict(X_test)
    xgb_pred = xgb_model.predict(X_test)

    # Evaluate models
    print_metrics(y_test, rf_pred, "Random Forest")
    print_metrics(y_test, xgb_pred, "XGBoost")

    # 🔥 User input using best model
    predict_user_input(xgb_model, scaler)


# 🔹 Run
if __name__ == "__main__":
    evaluate_ml_models()  