# td_zero.py

import numpy as np
from collections import defaultdict
from policies import epsilon_greedy_policy

np.random.seed(0)

def td_zero_control(env, num_episodes, gamma=0.9, alpha=0.3,
                    epsilon_start=0.2):
    n_actions = 4
    Q = defaultdict(lambda: np.zeros(n_actions))
    rewards_per_episode = []
    epsilon = epsilon_start

    for episode in range(num_episodes):
        state = env.reset()
        action = epsilon_greedy_policy(Q, state, n_actions, epsilon)
        done = False
        total_reward = 0

        while not done:
            next_state, reward, done = env.step(action)
            next_action = epsilon_greedy_policy(Q, next_state, n_actions, epsilon)
            td_target = reward + gamma * Q[next_state][next_action]
            td_error = td_target - Q[state][action]
            Q[state][action] += alpha * td_error
            state = next_state
            action = next_action
            total_reward += reward

        rewards_per_episode.append(total_reward)

    return Q, rewards_per_episode