import numpy as np
np.random.seed(42)

class GridWorldEnv:
    def __init__(self, random_start=False, max_steps=100):
        self.grid_size = 5
        self.start_pos = (0, 0)
        self.goal_pos = (4, 4)
        self.traps = [(0, 2), (1, 3), (3, 0), (3, 1)]
        self.rewards = {
            (1, 1): 5,    # 💰 گنج کوچک
            (2, 4): 5,    # 💰 گنج کوچک
            (4, 4): 10,   # 🏆 هدف
            (0, 2): -10,  # ☠️ تله‌ها
            (1, 3): -10,
            (3, 0): -10,
            (3, 1): -10
        }
        self.random_start = random_start
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        if self.random_start:
            free_cells = [(i, j) for i in range(self.grid_size)
                          for j in range(self.grid_size)
                          if (i, j) != self.goal_pos and (i, j) not in self.traps]
            self.agent_pos = free_cells[np.random.choice(len(free_cells))]
        else:
            self.agent_pos = self.start_pos
        self.steps = 0
        self.visited_rewards = set()  # ← خانه‌های با پاداش مثبت که قبلاً بازدید شده‌اند
        return self.get_state()

    def step(self, action):
        x, y = self.agent_pos

        if action == 0: x = max(x - 1, 0)
        elif action == 1: x = min(x + 1, self.grid_size - 1)
        elif action == 2: y = max(y - 1, 0)
        elif action == 3: y = min(y + 1, self.grid_size - 1)

        self.agent_pos = (x, y)
        self.steps += 1

        # ← پاداش فقط بار اول اعمال شود
        if self.agent_pos in self.rewards:
            if self.agent_pos in self.visited_rewards:
                reward = -1  # بار دوم یا بیشتر: فقط هزینه حرکت
            else:
                reward = self.rewards[self.agent_pos]
                self.visited_rewards.add(self.agent_pos)
        else:
            reward = -1

        done = (
            self.agent_pos in self.traps or
            self.agent_pos == self.goal_pos or
            self.steps >= self.max_steps
        )

        return self.get_state(), reward, done

    def get_state(self):
        return self.agent_pos[0] * self.grid_size + self.agent_pos[1]

    def render(self):
        grid = [[' ']*5 for _ in range(5)]
        for (x, y) in self.traps:
            grid[x][y] = '☠️'
        for (x, y), r in self.rewards.items():
            if r == 5:
                grid[x][y] = '$'
            elif r == 10:
                grid[x][y] = '🏆'
        ax, ay = self.agent_pos
        grid[ax][ay] = 'A'
        print("\n".join(["".join(f"{cell:^5}" for cell in row) for row in grid]))
        print()
