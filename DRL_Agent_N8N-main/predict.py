from stable_baselines3 import PPO
from n8n_agent.n8n_env import N8nWorkflowEnv

def corriger_workflow_n8n(workflow_path):
    # Instancie l'environnement avec le workflow à corriger
    env = N8nWorkflowEnv(workflow_path=workflow_path)
    # Charge le modèle entraîné
    model = PPO.load("ppo_n8n_agent.zip")
    # Reset l'environnement pour obtenir l'observation initiale
    obs, _ = env.reset()
    done = False
    steps = 0
    max_steps = env.max_steps

    # Boucle d'inférence : l'agent corrige le workflow étape par étape
    while not done and steps < max_steps:
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        steps += 1

    # À la fin, retourne l'état corrigé (ou tout autre info pertinente)
    # Ici, tu pourrais sauvegarder ou afficher le workflow modifié
    return env.state
