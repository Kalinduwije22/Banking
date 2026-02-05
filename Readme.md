# 🛡️ Real-Time Banking Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-GPU-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Kafka](https://img.shields.io/badge/Apache_Kafka-Streaming-231F20?style=for-the-badge&logo=apachekafka&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)

**An End-to-End MLOps pipeline that processes high-frequency financial transactions and detects fraud in milliseconds using a GPU-accelerated Neural Network.**

---

## 🚀 Project Overview

This application mimics a modern banking security infrastructure. Unlike traditional fraud systems that run "batch jobs" overnight, this system is **Event-Driven**. It processes transactions the moment they occur.

It simulates a live stream of credit card swipes, passes them through an **Apache Kafka** cluster, and analyzes them using a custom **PyTorch** model. The system instantly flags suspicious activity and visualizes the network traffic on a live **Streamlit Dashboard**.

### ✨ Key Features
* **⚡ Event-Driven Architecture:** Decoupled architecture using **Apache Kafka** ensures zero data loss during high-volume traffic.
* **🧠 Deep Learning "Brain":** A custom Neural Network (PyTorch) trained on imbalanced datasets to detect subtle fraud patterns.
* **🏎️ GPU Acceleration:** Optimized for NVIDIA CUDA to handle inference latency under 10ms per transaction.
* **📊 Real-Time Visualization:** A professional dashboard showing live transaction ticks, risk scores, and system health.
* **🛠️ Automated DevOps:** Fully containerized infrastructure with a "One-Command" setup using Makefile automation.

---

## 🛠️ Tech Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **AI Model** | **PyTorch (CUDA)** | Neural Network for binary classification |
| **Streaming** | **Apache Kafka** | High-throughput message broker |
| **Infrastructure** | **Docker & Zookeeper** | Containerized environment management |
| **Frontend** | **Streamlit** | Real-time interactive dashboard |
| **Tracking** | **MLflow** | Experiment tracking and model versioning |
| **Orchestration** | **Make / Prefect** | Pipeline automation and task flow |

---

## ⚡ Quick Start Guide

### 1. Prerequisites
* **Python 3.10** or higher installed.
* **Docker Desktop** installed and running.
* **NVIDIA GPU** (Optional, recommended for training speed).

### 2. Installation
Clone the repository and set up the environment using the automation Makefile.

```powershell
# 1. Setup Virtual Env & Install Dependencies
make setup

# 2. Start Kafka Infrastructure (Docker)
make infra-up

# 3. Train the AI Model
make train

# 4. Run the Full System (Requires 2 terminals)
# Terminal A: Start the Transaction Simulator
make stream

# Terminal B: Launch the Dashboard
make web

banking_fraud/
├── artifacts/              # Saved PyTorch models (.pth)
├── config/
│   └── config.yaml         # Central configuration (Hyperparams, Topics)
├── src/
│   ├── model.py            # Neural Network Architecture
│   ├── train.py            # Training Loop with MLflow
│   ├── inference.py        # Kafka Consumer & Predictor logic
│   ├── producer.py         # Transaction Simulator
│   └── orchestrate.py      # Automated Pipeline Flow
├── web/
│   └── app.py              # Streamlit Dashboard Source
├── docker-compose.yml      # Kafka & Zookeeper Infrastructure
├── Makefile                # Automation commands for setup & running
├── requirements.txt        # Python dependencies
└── .env                    # Environment Variables

⚠️ Disclaimer
This project is a simulation for educational and portfolio purposes. The transaction data is synthetic, and the fraud patterns are mathematically generated.