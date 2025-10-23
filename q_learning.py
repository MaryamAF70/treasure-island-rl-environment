# q_learning.py

import numpy as np
from collections import defaultdict
from policies import epsilon_greedy_policy

def q_learning(env, num_episodes, gamma=0.9, alpha=0.3,
               epsilon_start=0.2):
    np.random.seed(0)  # برای تکرارپذیری

    n_actions = 4  # بالا، پایین، چپ، راست
    Q = defaultdict(lambda: np.zeros(n_actions))
    rewards_per_episode = []

    epsilon = epsilon_start

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = epsilon_greedy_policy(Q, state, n_actions, epsilon)
            next_state, reward, done = env.step(action)

            max_next_q = np.max(Q[next_state])  # حداکثر Q در حالت بعدی

            # به‌روزرسانی Q
            Q[state][action] += alpha * (reward + gamma * max_next_q - Q[state][action])

            state = next_state
            total_reward += reward

        rewards_per_episode.append(total_reward)

    return Q, rewards_per_episode