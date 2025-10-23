import matplotlib.pyplot as plt
import numpy as np
np.random.seed(0)

# محاسبه میانگین تجمعی پاداش
def cumulative_average(data):
    return np.cumsum(data) / np.arange(1, len(data) + 1)

# استایل رسم برای الگوریتم‌ها
style_map = {
    'Monte Carlo':    {'color': 'black',     'marker': None, 'linestyle': '-'},
    'TD(0)':          {'color': 'dimgray',   'marker': None, 'linestyle': '--'},
    'TD(λ)':          {'color': 'gray',      'marker': 'x',  'linestyle': '-'},
    'Q-learning':     {'color': 'darkgray',  'marker': 'o',  'linestyle': '-'},
    'SARSA':          {'color': 'silver',    'marker': '^',  'linestyle': '-'},
}

# رسم نمودار میانگین تجمعی پاداش
def plot_rewards(reward_dict, sample_every=300):
    plt.figure(figsize=(24, 12))
    for label, rewards in reward_dict.items():
        averaged = cumulative_average(rewards)
        sampled_indices = np.arange(0, len(averaged), sample_every)
        sampled_rewards = averaged[sampled_indices]

        style = style_map.get(label, {'color': 'black', 'marker': None, 'linestyle': '-'})

        plt.plot(sampled_indices, sampled_rewards, label=label,
                 color=style['color'],
                 linestyle=style['linestyle'],
                 marker=style['marker'] if style['marker'] else None,
                 linewidth=2.5,
                 markersize=8,
                 markeredgewidth=2,
                 markeredgecolor=style['color'],
                 alpha=1)

    plt.title("Cumulative Average Reward per Episode", fontsize=20)
    plt.xlabel("Episode", fontsize=20)
    plt.ylabel("Cumulative Average Reward", fontsize=20)
    plt.legend(fontsize=15)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.show()

# رسم زمان اجرا الگوریتم‌ها
def plot_times(runtime_dict):
    plt.figure(figsize=(10, 6))
    names = list(runtime_dict.keys())
    times = list(runtime_dict.values())

    bars = plt.bar(names, times, color='gray', edgecolor='black', hatch='///')
    plt.title('Execution Time Comparison', fontsize=22, fontweight='bold')
    plt.ylabel('Time (seconds)', fontsize=20, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=1)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.show()


# نمایش مسیر یادگرفته‌شده
def show_learned_path(env, Q, title="Learned Path"):
    state = env.reset()
    env.agent_pos = env.start_pos
    path = [env.agent_pos]
    visited = set()

    for _ in range(env.max_steps):
        s = env.get_state()
        if s not in Q:
            break
        a = np.argmax(Q[s])
        _, _, done = env.step(a)
        path.append(env.agent_pos)
        if done or env.agent_pos in visited:
            break
        visited.add(env.agent_pos)

    grid = [[' ']*env.grid_size for _ in range(env.grid_size)]
    for (x, y) in env.traps:
        grid[x][y] = '☠️'
    for (x, y), r in env.rewards.items():
        if r == 5:
            grid[x][y] = '$'
        elif r == 10:
            grid[x][y] = '🏆'

    for i, (x, y) in enumerate(path):
        grid[x][y] = 'A' if i == 0 else '·'

    print(f"\n{title}")
    print("\n".join(["".join(f"{cell:^5}" for cell in row) for row in grid]))
    print()