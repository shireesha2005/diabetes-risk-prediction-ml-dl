import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from src.preprocess import load_data, preprocess_data


# 🔹 SAME DNN ARCHITECTURE (must match training)
class DiabetesDNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# 🔹 Evaluation Function
def evaluate_dnn():
    # Load data
    df = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    # Convert test data to tensor
    X_test = torch.tensor(X_test, dtype=torch.float32)

    # Load model (safe loading)
    model = DiabetesDNN(X_test.shape[1])
    model.load_state_dict(torch.load("dnn_model.pth", map_location=torch.device('cpu')))

    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        probs = outputs.numpy()
        predictions = (probs > 0.5).astype(int)

    # 🔹 Debug info (important)
    print("Unique Predictions:", set(predictions.flatten()))

    # 🔹 Metrics
    print("\n--- DNN Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")
    print(f"Precision: {precision_score(y_test, predictions):.2f}")
    print(f"Recall: {recall_score(y_test, predictions):.2f}")
    print(f"F1 Score: {f1_score(y_test, predictions):.2f}")

    # 🔹 Confusion Matrix Visualization
    cm = confusion_matrix(y_test, predictions)

    plt.figure()
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=["Not Diabetic", "Diabetic"],
        yticklabels=["Not Diabetic", "Diabetic"]
    )
    plt.title("DNN Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


# 🔹 Run Evaluation
if __name__ == "__main__":
    evaluate_dnn()   