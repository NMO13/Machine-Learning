from mouse_world import MouseWorld
import tensorflow as tf      # Deep Learning library
from neural_network.my_neural_network import MyNeuralNet
import random
import numpy as np
from collections import deque# Ordered collection with ends

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

memory_size = 1000000          # Number of experiences the Memory can keep
batch_size = 64
pretrain_length = batch_size
n_neurons_in_h1 = 60
n_neurons_in_h2 = 60

class DQNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='DQNetwork'):
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.actions_ = tf.placeholder(tf.float32, [None, action_size], name="actions_")
        # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
        self.target_Q = tf.placeholder(tf.float32, [None], name="target")
        n_features = state_size
        n_classes = action_size

        # placeholdr tensors built to store features(in X) , labels(in Y) and dropout probability(in keep_prob)
        self.X = tf.placeholder(tf.float32, [None, n_features], name='features')
        self.Y = tf.placeholder(tf.float32, [None, n_classes], name='labels')
        keep_prob = tf.placeholder(tf.float32, name='drop_prob')

        W1 = tf.Variable(tf.truncated_normal([n_features, n_neurons_in_h1], mean=0, stddev=1 / np.sqrt(n_features)),
                         name='weights1')
        b1 = tf.Variable(tf.truncated_normal([n_neurons_in_h1], mean=0, stddev=1 / np.sqrt(n_features)), name='biases1')
        y1 = tf.nn.tanh((tf.matmul(self.X, W1) + b1), name='activationLayer1')

        # network parameters(weights and biases) are set and initialized(Layer2)
        W2 = tf.Variable(tf.random_normal([n_neurons_in_h1, n_neurons_in_h2], mean=0, stddev=1 / np.sqrt(n_features)),
                         name='weights2')
        b2 = tf.Variable(tf.random_normal([n_neurons_in_h2], mean=0, stddev=1 / np.sqrt(n_features)), name='biases2')
        # activation function(sigmoid)
        y2 = tf.nn.sigmoid((tf.matmul(y1, W2) + b2), name='activationLayer2')

        # output layer weights and biasies
        Wo = tf.Variable(tf.random_normal([n_neurons_in_h2, n_classes], mean=0, stddev=1 / np.sqrt(n_features)),
                         name='weightsOut')
        bo = tf.Variable(tf.random_normal([n_classes], mean=0, stddev=1 / np.sqrt(n_features)), name='biasesOut')
        # activation function(softmax)
        self.output = tf.nn.sigmoid((tf.matmul(y2, Wo) + bo), name='activationOutputLayer')

        self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)

        self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))

        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
        # cost function
        #l = tf.losses.mean_squared_error(self.Y, self.output)
        # optimizer
        #train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(l)

class Memory():
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        if action == 'U':
            action = [1, 0, 0, 0]
        if action == 'D':
            action = [0, 1, 0, 0]
        if action == 'L':
            action = [0, 0, 1, 0]
        if action == 'R':
            action = [0, 0, 0, 1]
        self.buffer.append((state, action, reward, next_state, done))

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
        ### MODEL HYPERPARAMETERS
        state_size = 2  # Our input is a stack of 4 frames hence 84x84x4 (Width, height, channels)
        action_size = 4
        learning_rate = 0.0002  # Alpha (aka learning rate)
        # Reset the graph
        tf.reset_default_graph()

        # Instantiate the DQNetwork
        self.net = DQNetwork(state_size, action_size, learning_rate)
        super().__init__(width, height, local_rewards)

    def get_action(self, explore_start, explore_stop, decay_rate, decay_step):
        number = np.random.uniform()
        eps = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
        action = None
        if number <= eps:
            # explore
            action = random.choice(list(self.Q[self.current_state].keys()))
            return action
        else:
            # exploit
            stats = sess.run(self.net.output, feed_dict={mw.net.X: np.array([self.current_state])})
            stats = stats[0]
            possible_states = list(self.Q[self.current_state].keys())
            if 'U' not in possible_states:
                stats[0] = -10000
            if 'D' not in possible_states:
                stats[1] = -10000
            if 'L' not in possible_states:
                stats[2] = -10000
            if 'R' not in possible_states:
                stats[3] = -10000
            action = np.argmax(stats)
            if action == 0:
                return 'U'
            elif action == 1:
                return 'D'
            elif action == 2:
                return 'L'
            else:
                return 'R'

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

    def pretrain(self, memory):
        mw.current_state = (0, 0)
        for i in range(pretrain_length):
            # If it's the first step
            if i == 0:
                # First we need a state
                state = self.current_state

            # Random action
            action = self.get_action(1, 0, 0, 0)

            # Get the rewards
            reward = self.make_action(action)

            # Look if the episode is finished
            done = self.game_over()

            # If we're dead
            if done:
                # We finished the episode
                next_state = (-1, -1)

                # Add experience to memory
                memory.add(state, action, reward, next_state, done)

                # Start a new episode
                mw.current_state = (0, 0)

                # First we need a state
                state = self.current_state
            else:
                # Get the next state
                next_state = self.current_state

                # Add experience to memory
                memory.add(state, action, reward, next_state, done)

                # Our state is now the next_state
                state = next_state

    def get_random_state(self):
        import random
        res = random.choice(list(self.Q.keys()))
        return res

    def print_best_policy(self):
        for i in range(self.height):
            print("---------------------------")
            for j in range(self.width):
                stats = self.Q[(i, j)]
                action = max(stats.keys(), key=(lambda key: stats[key]))
                print("  %s  |" % action, end="")
            print("")

if __name__ == "__main__":
    import time
    start = time.time()
    mw = MouseWorld2(5, 5, {(0, 1): 0, (1, 0): 0, (4, 4): 1, (1, 1): -1, (3, 4): -1, (0, 2): -1})
    mw.print_values(mw.rewards)
    # Q learning hyperparameters
    gamma = 0.95  # Discounting rate

    explore_start = 1.0  # exploration probability at start
    explore_stop = 0.01  # minimum exploration probability
    decay_rate = 0.0001

    mw.pretrain(memory)
    with tf.Session() as sess:
        # Initialize the variables
        sess.run(tf.global_variables_initializer())

        decay_step = 0
        for episode in range(700):
            print('Episode: ' + str(episode))
            # Initialize the rewards of the episode
            episode_rewards = []
            # reset mouse
            mw.current_state = mw.get_random_state()

            while not mw.game_over():
                # Increase decay_step
                decay_step += 1

                state = mw.current_state
                action = mw.get_action(explore_start, explore_stop, decay_rate, decay_step)
                reward = mw.make_action(action)
                # Add the reward to total reward
                episode_rewards.append(reward)
                next_state = mw.current_state

                if mw.game_over():
                    memory.add(state, action, reward, (-1, -1), True)
                else:
                    memory.add(state, action, reward, next_state, False)

            batch = memory.sample(batch_size)
            states_mb = np.array([each[0] for each in batch], ndmin=3)
            actions_mb = np.array([each[1] for each in batch])
            rewards_mb = np.array([each[2] for each in batch])
            next_states_mb = np.array([each[3] for each in batch], ndmin=3)
            dones_mb = np.array([each[4] for each in batch])

            # Get Q values for next_state
            Qs_next_state = sess.run(mw.net.output, feed_dict={mw.net.X: next_states_mb[0]})

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
            cur_state_mb = states_mb
            sess.run([mw.net.optimizer],
                     feed_dict={mw.net.X: states_mb[0],
                                mw.net.target_Q: targets_mb,
                                mw.net.actions_: actions_mb})

        for k, v in mw.Q.items():
            stats = sess.run(mw.net.output, feed_dict={mw.net.X: np.array([k])})
            stats = stats[0]
            max_val = np.argmax(stats)
            q_state =  mw.Q[k]
            if max_val == 0:
               q_state['U'] = 1.0
            if max_val == 1:
               q_state['D'] = 1.0
            if max_val == 2:
               q_state['L'] = 1.0
            if max_val == 3:
               q_state['R'] = 1.0
    end = time.time()
    mw.print_best_policy()
    print('Finished')
    print('time: {}'.format(end - start))