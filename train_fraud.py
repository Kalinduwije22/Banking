import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.pytorch

# 1. Setup Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Generate Synthetic "Banking" Data
# 10,000 transactions, only 2% are fraud (weights=0.98, 0.02)
print("Generating banking data...")
X, y = make_classification(n_samples=10000, n_features=10, n_classes=2, weights=[0.98, 0.02], random_state=42)
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Convert to Tensors
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

# 3. Define Model
class FraudModel(nn.Module):
    def __init__(self):
        super(FraudModel, self).__init__()
        self.layer1 = nn.Linear(10, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.layer2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.output(x))
        return x

# 4. MLflow Experiment Setup
mlflow.set_experiment("Banking_Fraud_Detection")

def train_model(learning_rate, epochs):
    with mlflow.start_run():
        # Log Parameters (The "Check Parameters" part)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("model_type", "FeedForward_NN")

        model = FraudModel().to(device)
        criterion = nn.BCELoss() # Binary Cross Entropy
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        print(f"Training with LR={learning_rate}...")
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

        # Evaluate
        with torch.no_grad():
            test_outputs = model(X_test)
            predicted = (test_outputs > 0.5).float()
            accuracy = (predicted == y_test).float().mean()
            
            # Log Metrics (The Results)
            mlflow.log_metric("final_loss", loss.item())
            mlflow.log_metric("accuracy", accuracy.item())
            
            print(f"Run Finished -> Accuracy: {accuracy.item():.4f}")

        # Save Model for Inference
        torch.save(model.state_dict(), "fraud_model.pth")
        mlflow.pytorch.log_model(model, "model")

# Run the training
if __name__ == "__main__":
    train_model(learning_rate=0.001, epochs=200)