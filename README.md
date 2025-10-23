# Treasure Island Environment

This project provides a custom reinforcement learning environment designed to evaluate and compare fundamental RL algorithms in a simple yet instructive setup.

# Environment Description

The environment is a 5×5 grid world called Treasure Island.

The agent (A) starts at the top-left corner (0, 0).

The goal (🏆) is located at the bottom-right corner (4, 4) and gives a large positive reward when reached.

Positive rewards (💰) encourage exploration but do not end the episode.

Traps (☠️) give a negative reward and terminate the episode.

The agent can move in four directions: up, down, left, and right.

The environment is implemented in Python as the class GridWorldEnv, including methods for reset(), step(), get_state(), and render().

# Algorithms Implemented

The environment has been used to test and compare the following reinforcement learning algorithms:

Monte Carlo

TD(0)

TD(λ)

Q-Learning

SARSA

# Project Goal

Compare learning speed, cumulative rewards, and execution time of RL algorithms under identical conditions.

Analyze each algorithm’s behavior in response to rewards and traps.

Study the effect of RL parameters (α, γ, ε) in ε-greedy policies.
