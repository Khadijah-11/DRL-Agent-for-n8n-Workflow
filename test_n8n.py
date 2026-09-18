from n8n_agent.n8n_env import N8nWorkflowEnv

env = N8nWorkflowEnv("ai_workflow.json")
obs, _ = env.reset()
done = False

while not done:
    env.render()
    action = env.action_space.sample()
    obs, reward, done, _, _ = env.step(action)
    print(f"Action: {action}, Reward: {reward}")
