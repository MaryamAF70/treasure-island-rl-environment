# monte_carlo.py

import numpy as np
from collections import defaultdict
from policies import epsilon_greedy_policy

np.random.seed(0)

def monte_carlo_control(env, num_episodes, gamma=0.9,
                         epsilon_start=0.2):
    n_actions = 4
    Q = defaultdict(lambda: np.zeros(n_actions))
    N = defaultdict(lambda: np.zeros(n_actions))  # برای میانگین افزایشی
    rewards_per_episode = []
    epsilon = epsilon_start

    for episode in range(num_episodes):
        state = env.reset()
        episode_data = []
        done = False
        total_reward = 0

        while not done:
            action = epsilon_greedy_policy(Q, state, n_actions, epsilon)
            next_state, reward, done = env.step(action)
            episode_data.append((state, action, reward))
            total_reward += reward
            state = next_state

        G = 0
        visited = set()
        for t in reversed(range(len(episode_data))):
            s, a, r = episode_data[t]
            G = gamma * G + r
            if (s, a) not in visited:
                visited.add((s, a))
                N[s][a] += 1
                Q[s][a] += (G - Q[s][a]) / N[s][a]  # میانگین افزایشی

        rewards_per_episode.append(total_reward)

    return Q, rewards_per_episode