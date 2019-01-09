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

    def print_values(self):
        for i in range(self.width):
            print("---------------------------")
            for j in range(self.height):
                v = self.rewards.get((i, j), 0)
                if v >= 0:
                    print(" %.2f|" % v, end="")
                else:
                    print("%.2f|" % v, end="")  # -ve sign takes up an extra space
            print("")

    def game_over(self):
        return self.current_state == (1, 1) or self.current_state == (1, 2)

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
        self.Q[self.current_state][action] = Q_s_a + self.alpha *(self.rewards[next_state] + self.gamma*(max_Q_prime - Q_s_a))
        self.current_state = next_state


if __name__ == "__main__":
    mw = MouseWorld(2, 3, {(0, 1): 3, (1, 0): 3, (1, 2): 10, (1, 1): -3})
    mw.print_values()
    for i in range(10000):
        mw.current_state = (0, 0)
        while not mw.game_over():
            mw.move()
    print('Finished')
