import torch
import torch.nn as nn
from kafka import KafkaConsumer
import json
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# Load the model
model = FraudModel().to(device)
model.load_state_dict(torch.load("fraud_model.pth"))
model.eval()

consumer = KafkaConsumer(
    'banking_transactions',
    bootstrap_servers=['localhost:29092'],
    auto_offset_reset='latest',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

print("🛡️ Banking Security System Active...")

with torch.no_grad():
    for message in consumer:
        data = message.value
        features = torch.tensor(data['features'], dtype=torch.float32).to(device)
        
        # Predict Probability
        output = model(features.unsqueeze(0))
        fraud_prob = output.item()
        
        # Rule: If probability > 80%, Block it.
        status = "✅ APPROVED"
        if fraud_prob > 0.80:
            status = "🚨 FRAUD DETECTED (BLOCKED)"
            
        print(f"Trans #{data['trans_id']} | Risk Score: {fraud_prob:.4f} | {status}")