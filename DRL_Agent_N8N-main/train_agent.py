from n8n_agent.n8n_env import N8nWorkflowEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure

# Créer l'environnement
env = N8nWorkflowEnv()

# Configurer le logger
log_dir = "./logs/"
new_logger = configure(log_dir, ["stdout", "tensorboard"])

# Créer le modèle
model = PPO("MlpPolicy", env, verbose=1)
model.set_logger(new_logger)  # Le logger doit être défini avant le learn()

# Entraîner le modèle
model.learn(total_timesteps=10000)

# Sauvegarder le modèle
model.save("ppo_n8n_agent")
