import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from imblearn.over_sampling import SMOTE
from src.preprocess import load_data, preprocess_data


# 🔹 Deep Neural Network
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


# 🔹 Training Function
def train_dnn():
    # Load data
    df = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    # 🔥 Apply SMOTE (only on training data)
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

    # 🔥 Create DataLoader (mini-batch training)
    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Model
    model = DiabetesDNN(X_train.shape[1])

    # Loss & optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0003)

    epochs = 100

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_X, batch_y in dataloader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Print average loss
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), "dnn_model.pth")

    print("\n✅ DNN Model trained with SMOTE + Batch Training!")


# Run
if __name__ == "__main__":
    train_dnn()
