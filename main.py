import time
from island_env import GridWorldEnv
from monte_carlo import monte_carlo_control
from td_zero import td_zero_control
from td_lambda import td_lambda_control
from q_learning import q_learning
from sarsa import sarsa
from plot_utils import (
    plot_rewards,
    plot_times,
    show_learned_path
)

# نمایش اولیه محیط
env = GridWorldEnv()
env.render()

# تعداد اپیزود برای آموزش
num_episodes = 10000

# لیست برای ذخیره پاداش‌ها و Qها
reward_dict = {}
q_tables = {}
times = {}

# ---------------- Monte Carlo ----------------
print("Training Monte Carlo...")
start = time.time()
Q_mc, rewards_mc = monte_carlo_control(env, num_episodes)
elapsed = time.time() - start
reward_dict['Monte Carlo'] = rewards_mc
q_tables['Monte Carlo'] = Q_mc
times['Monte Carlo'] = elapsed
print(f"Monte Carlo finished in {elapsed:.2f} seconds")
show_learned_path(env, Q_mc, title="Monte Carlo Learned Path")

# ---------------- TD(0) ----------------
print("Training TD(0)...")
start = time.time()
Q_td0, rewards_td0 = td_zero_control(env, num_episodes)
elapsed = time.time() - start
reward_dict['TD(0)'] = rewards_td0
q_tables['TD(0)'] = Q_td0
times['TD(0)'] = elapsed
print(f"TD(0) finished in {elapsed:.2f} seconds")
show_learned_path(env, Q_td0, title="TD(0) Learned Path")

# ---------------- TD(λ) ----------------
print("Training TD(λ)...")
start = time.time()
Q_td_lambda, rewards_td_lambda = td_lambda_control(env, num_episodes, lam=0.8)
elapsed = time.time() - start
reward_dict['TD(λ)'] = rewards_td_lambda
q_tables['TD(λ)'] = Q_td_lambda
times['TD(λ)'] = elapsed
print(f"TD(λ) finished in {elapsed:.2f} seconds")
show_learned_path(env, Q_td_lambda, title="TD(λ) Learned Path")

# ---------------- Q-learning ----------------
print("Training Q-learning...")
start = time.time()
Q_ql, rewards_ql = q_learning(env, num_episodes)
elapsed = time.time() - start
reward_dict['Q-learning'] = rewards_ql
q_tables['Q-learning'] = Q_ql
times['Q-learning'] = elapsed
print(f"Q-learning finished in {elapsed:.2f} seconds")
show_learned_path(env, Q_ql, title="Q-learning Learned Path")

# ---------------- SARSA ----------------
print("Training SARSA...")
start = time.time()
Q_sarsa, rewards_sarsa = sarsa(env, num_episodes)
elapsed = time.time() - start
reward_dict['SARSA'] = rewards_sarsa
q_tables['SARSA'] = Q_sarsa
times['SARSA'] = elapsed
print(f"SARSA finished in {elapsed:.2f} seconds")
show_learned_path(env, Q_sarsa, title="SARSA Learned Path")

# ---------------- نمودارها ----------------
plot_rewards(reward_dict)             # میانگین تجمعی پاداش‌ها
plot_times(times)                     # زمان اجرا