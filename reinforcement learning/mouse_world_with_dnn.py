from mouse_world import MouseWorld
from neural_network.my_neural_network import MyNeuralNet
import random
import numpy as np
from collections import deque# Ordered collection with ends
memory_size = 1000000          # Number of experiences the Memory can keep
batch_size = 64

class Memory():
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                 size=batch_size,
                                 replace=False)

        return [self.buffer[i] for i in index]


# Instantiate memory
memory = Memory(max_size=memory_size)

class MouseWorld2(MouseWorld):

    def __init__(self, width, height, local_rewards):
        self.net = MyNeuralNet()
        super().__init__(width, height, local_rewards)

    def get_action(self, eps):
        number = np.random.uniform()
        action = None
        if number <= eps:
            # explore
            action = random.choice(self.actions)
        else:
            # exploit
            stats = self.Q[self.current_state]
            action = max(stats.keys(), key=(lambda key: stats[key]))
        return action

    def make_action(self, action):
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

        self.current_state = (i, j)
        return self.rewards[self.current_state]

    def move(self, eps):
        pass

if __name__ == "__main__":
    import time
    start = time.time()
    mw = MouseWorld2(5, 5, {(0, 1): 0, (1, 0): 0, (4, 4): 1, (1, 1): -1, (3, 4): -1, (0, 2): -1})
    mw.print_values(mw.rewards)
    # Exploration parameters
    epsilon = 0.3  # Exploration rate
    max_epsilon = 1.0  # Exploration probability at start
    min_epsilon = 0.01  # Minimum exploration probability
    decay_rate = 0.005  # Exponential decay rate for exploration prob
    # Q learning hyperparameters
    gamma = 0.95  # Discounting rate
    for episode in range(10000):
        # Initialize the rewards of the episode
        episode_rewards = []
        # reset mouse
        mw.current_state = (0, 0)
        while not mw.game_over():
            state = mw.current_state
            action = mw.get_action(epsilon)
            reward = mw.make_action(action)
            # Add the reward to total reward
            episode_rewards.append(reward)
            next_state = mw.current_state

            if mw.game_over():
                memory.add((state, action, reward, None, False))
            else:
                memory.add((state, action, reward, next_state, False))

        batch = memory.sample(batch_size)
        states_mb = np.array([each[0] for each in batch], ndmin=3)
        actions_mb = np.array([each[1] for each in batch])
        rewards_mb = np.array([each[2] for each in batch])
        next_states_mb = np.array([each[3] for each in batch], ndmin=3)
        dones_mb = np.array([each[4] for each in batch])

        Qs_next_state = mw.net.classify(next_states_mb)

        target_Qs_batch = []

        for i in range(0, len(batch)):
            terminal = dones_mb[i]

            # If we are in a terminal state, only equals reward
            if terminal:
                target_Qs_batch.append(rewards_mb[i])

            else:
                target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
                target_Qs_batch.append(target)

        targets_mb = np.array([each for each in target_Qs_batch])

    end = time.time()
    mw.print_best_policy()
    print('Finished')
    print('time: {}'.format(end - start))