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

DRL-Agent-for-n8n-Workflow/

│── app.py                 # Streamlit UI
│── train_agent.py         # DRL agent training loop
│── predict.py              # Run predictions with a trained agent
│── test_n8n.py               # Test script for n8n workflow interaction
│── n8n_agent/               # DRL agent implementation
│── ppo_n8n_agent.zip      # Trained PPO model
│── ai_workflow.json         # Example n8n workflow
│── sample_workflow.json  # Example n8n workflow
│── requirements.txt        # Python dependencies
│── README.md                # Project documentation

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



https://github.com/user-attachments/assets/f3826280-87d5-4124-9a6e-621346383f50



## 📊 Training with PPO

We use Proximal Policy Optimization (PPO) from Stable-Baselines3:

Reward function encourages valid workflow corrections

Logging and evaluation are tracked via TensorBoard

Training scripts are fully configurable in train.py


## 👥 Authors

Khadija Tagui &
Nisrin Lasfer
