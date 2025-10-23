# policies.py

import random
import numpy as np
random.seed(0)
def epsilon_greedy_policy(Q, state, n_actions, epsilon=0.1):
    if random.random() < epsilon:
        return np.random.randint(n_actions)
    else:
        return np.argmax(Q[state])