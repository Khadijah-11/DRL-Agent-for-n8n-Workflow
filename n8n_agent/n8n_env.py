import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json

class N8nWorkflowEnv(gym.Env):
    def __init__(self, workflow_path="sample_workflow.json"):
        self.workflow_path = workflow_path
        self.max_nodes = 10  # Pad/truncate to fixed size
        self.current_step = 0
        self.max_steps = 20

        # [type, has_credentials, disabled, outgoing, incoming, is_broken]
        self.feature_size = 6
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.max_nodes, self.feature_size), dtype=np.float32)
        self.action_space = spaces.Discrete(self.max_nodes)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.state = self._load_workflow_state()
        return self.state, {}

    def _load_workflow_state(self):
        with open(self.workflow_path) as f:
            workflow = json.load(f)

        node_features = []

        node_name_map = {node["name"]: node for node in workflow["nodes"]}

        # Build incoming/outgoing map
        incoming_map = {node["name"]: 0 for node in workflow["nodes"]}
        outgoing_map = {node["name"]: 0 for node in workflow["nodes"]}
        for src, conns in workflow.get("connections", {}).items():
            for out_list in conns.get("main", []):
                for conn in out_list:
                    outgoing_map[src] += 1
                    incoming_map[conn["node"]] += 1

        for node in workflow["nodes"]:
            typ = node["type"]
            node_vec = [
                self._encode_type(typ),                              # type
                1 if node.get("credentials") else 0,                 # has_credentials
                1 if node.get("disabled", False) else 0,             # disabled
                outgoing_map[node["name"]],                          # outgoing
                incoming_map[node["name"]],                          # incoming
                1 if "broken" in node.get("name", "").lower() else 0 # is_broken (simulated)
            ]
            node_features.append(node_vec)

        # Pad or truncate to fixed size
        padded = node_features[:self.max_nodes]
        while len(padded) < self.max_nodes:
            padded.append([0] * self.feature_size)

        return np.array(padded, dtype=np.float32)

    def _encode_type(self, typ):
        # Dummy encoder: assign a unique ID per type (you can use dict later)
        return hash(typ) % 10 / 10.0

    def step(self, action):
        self.current_step += 1

        reward = 0
        done = False

        if self.state[action][5] == 1:  # If is_broken
            self.state[action][5] = 0   # Mark as fixed
            reward = 1

        if np.sum(self.state[:, 5]) == 0:
            done = True
            reward += 5

        if self.current_step >= self.max_steps:
            done = True

        return self.state, reward, done, False, {}

    def render(self):
        print(f"Step {self.current_step}")
        for i, node in enumerate(self.state):
            print(f"Node {i}: {node}")
