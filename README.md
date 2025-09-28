# 🧠 DRL Agent for n8n Workflow Correction

This project implements a Deep Reinforcement Learning (DRL) agent designed to automatically detect and correct errors in n8n workflows.
It combines TensorFlow, Stable-Baselines3 (PPO), and Streamlit to provide both backend training and an interactive frontend interface.

## 🚀 Features

✅ Reinforcement Learning (PPO) agent trained on workflow correction tasks

✅ TensorBoard integration for monitoring training performance

✅ Streamlit interface for interactive testing and visualization

✅ n8n workflow parsing and fixing with automated correction suggestions

✅ Modular code structure for easy customization and extension

## 📂 Project Structure
DRL_Agent_N8N-main/
│── app.py              # Streamlit UI  
│── train.py            # DRL agent training loop  
│── agent/              # DRL agent implementation  
│── workflows/          # Example n8n workflows (training/test data)  
│── logs/               # TensorBoard logs  
│── requirements.txt    # Python dependencies  
│── README.md           # Project documentation  

## ⚡ Quickstart
### 1️⃣ Create and activate a virtual environment
python -m venv .env
source .env/bin/activate   # (Linux/Mac)
.env\Scripts\activate      # (Windows)

### 2️⃣ Install dependencies
pip install -r requirements.txt

### 3️⃣ Run TensorBoard (monitor training)
tensorboard --logdir=./logs/ --port=6006


Then open: http://localhost:6006

### 4️⃣ Launch the Streamlit app
streamlit run app.py



## 🎥 Demo

Here’s a walkthrough of the Streamlit interface:

https://github.com/user-attachments/assets/69915613-dea4-4088-83b4-d40f5cf2a2c4

## 📊 Training with PPO

We use Proximal Policy Optimization (PPO) from Stable-Baselines3:

Reward function encourages valid workflow corrections

Logging and evaluation are tracked via TensorBoard

Training scripts are fully configurable in train.py


## 👥 Authors

Khadija Tagui &
Nisrin Lasfer
