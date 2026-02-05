import time
import json
import random
from kafka import KafkaProducer
import numpy as np

producer = KafkaProducer(
    bootstrap_servers=['localhost:29092'],
    value_serializer=lambda x: json.dumps(x).encode('utf-8')
)

print("💳 Simulating Credit Card Swipes...")

# Infinite loop of transactions
transaction_id = 1000
while True:
    # Create fake feature vector (10 numbers representing location, amount, time...)
    fake_features = np.random.randn(10).tolist()
    
    # Randomly inject a "Fraud" pattern (extreme values)
    is_fraud_simulation = random.random() < 0.05 # 5% chance
    if is_fraud_simulation:
        fake_features = [x * 5 for x in fake_features] # Anomalous data

    payload = {
        'trans_id': transaction_id,
        'features': fake_features,
        'timestamp': time.time()
    }
    
    producer.send('banking_transactions', value=payload)
    print(f"Sent Transaction #{transaction_id}")
    
    transaction_id += 1
    time.sleep(0.1)