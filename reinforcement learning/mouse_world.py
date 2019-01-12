import numpy as np
import random
class MouseWorld:
    def __init__(self, width, height, local_rewards):
        self.width = width
        self.height = height
        self.alpha = 0.1
        self.gamma = 0.9

        self.actions = ['U', 'D', 'L', 'R']

        self.rewards = {}
        self.Q = {}
        for i in range(width):
            for j in range(height):
                self.rewards[(i, j)] = local_rewards.get((i, j), 0)
                self.Q[(i, j)] = {}
                for action in self.actions:
                    self.Q[(i, j)][action] = 0

    def print_values(self, values):
        for i in range(self.width):
            print("---------------------------")
            for j in range(self.height):
                v = values.get((i, j), 0)
                if v >= 0:
                    print(" %.2f|" % v, end="")
                else:
                    print("%.2f|" % v, end="")  # -ve sign takes up an extra space
            print("")

    def print_best_policy(self):
        for i in range(self.width):
            print("---------------------------")
            for j in range(self.height):
                stats = self.Q[(i, j)]
                action = max(stats.keys(), key=(lambda key: stats[key]))
                print("  %s  |" % action, end="")
            print("")

    def game_over(self):
        return self.current_state == (1, 1) or self.current_state == (1, 2)\
               or self.current_state == (3, 4) or self.current_state == (0, 2)

    def move(self):
        eps = 0.3
        number = np.random.uniform()
        action = None
        if number <= eps:
            # explore
            action = random.choice(self.actions)
        else:
            # exploit
            stats = self.Q[self.current_state]
            action = max(stats.keys(), key=(lambda key: stats[key]))

        i = self.current_state[0]
        j = self.current_state[1]
        if action == 'L':
            j -= 1
        if action == 'R':
            j += 1
        if action == 'U':
            i -= 1
        if action == 'D':
            i += 1

        next_state = (i, j)
        try:
            stats = self.Q[next_state]
        except:
            return
        max_Q_prime = max(stats.values())
        Q_s_a = self.Q[self.current_state][action]
        self.Q[self.current_state][action] = Q_s_a + self.alpha *(self.rewards[next_state] + self.gamma * max_Q_prime - Q_s_a)
        self.current_state = next_state


if __name__ == "__main__":
    import time
    start = time.time()
    mw = MouseWorld(5, 5, {(0, 1): 0, (1, 0): 0, (4, 4): 1, (1, 1): -1, (3, 4): -1, (0, 2): -1})
    mw.print_values(mw.rewards)
    # Exploration parameters
    epsilon = 1.0  # Exploration rate
    max_epsilon = 1.0  # Exploration probability at start
    min_epsilon = 0.01  # Minimum exploration probability
    decay_rate = 0.005  # Exponential decay rate for exploration prob
    for episode in range(10000):
        mw.current_state = (0, 0)
        while not mw.game_over():
            mw.move()
        # Reduce epsilon (because we need less and less exploration)
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    end = time.time()
    mw.print_best_policy()
    print('Finished')
    print('time: {}'.format(end - start))
