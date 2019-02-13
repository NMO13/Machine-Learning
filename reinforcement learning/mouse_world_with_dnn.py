from mouse_world import MouseWorld
import tensorflow as tf      # Deep Learning library
from neural_network.my_neural_network import MyNeuralNet
import random
import numpy as np
from collections import deque# Ordered collection with ends
memory_size = 1000000          # Number of experiences the Memory can keep
batch_size = 64
pretrain_length = batch_size

class DQNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            # We create the placeholders
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
            # [None, 84, 84, 4]
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, 3], name="actions_")

            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")

            """
            First convnet:
            CNN
            BatchNormalization
            ELU
            """
            # Input is 84x84x4
            self.conv1 = tf.layers.conv2d(inputs=self.inputs_,
                                          filters=32,
                                          kernel_size=[8, 8],
                                          strides=[4, 4],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv1")

            self.conv1_batchnorm = tf.layers.batch_normalization(self.conv1,
                                                                 training=True,
                                                                 epsilon=1e-5,
                                                                 name='batch_norm1')

            self.conv1_out = tf.nn.elu(self.conv1_batchnorm, name="conv1_out")
            ## --> [20, 20, 32]

            """
            Second convnet:
            CNN
            BatchNormalization
            ELU
            """
            self.conv2 = tf.layers.conv2d(inputs=self.conv1_out,
                                          filters=64,
                                          kernel_size=[4, 4],
                                          strides=[2, 2],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv2")

            self.conv2_batchnorm = tf.layers.batch_normalization(self.conv2,
                                                                 training=True,
                                                                 epsilon=1e-5,
                                                                 name='batch_norm2')

            self.conv2_out = tf.nn.elu(self.conv2_batchnorm, name="conv2_out")
            ## --> [9, 9, 64]

            """
            Third convnet:
            CNN
            BatchNormalization
            ELU
            """
            self.conv3 = tf.layers.conv2d(inputs=self.conv2_out,
                                          filters=128,
                                          kernel_size=[4, 4],
                                          strides=[2, 2],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv3")

            self.conv3_batchnorm = tf.layers.batch_normalization(self.conv3,
                                                                 training=True,
                                                                 epsilon=1e-5,
                                                                 name='batch_norm3')

            self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name="conv3_out")
            ## --> [3, 3, 128]

            self.flatten = tf.layers.flatten(self.conv3_out)
            ## --> [1152]

            self.fc = tf.layers.dense(inputs=self.flatten,
                                      units=512,
                                      activation=tf.nn.elu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      name="fc1")

            self.output = tf.layers.dense(inputs=self.fc,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          units=3,
                                          activation=None)

            # Q is our predicted Q value.
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)

            # The loss is the difference between our predicted Q_values and the Q_target
            # Sum(Qtarget - Q)^2
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))

            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

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
        ### MODEL HYPERPARAMETERS
        state_size = [84, 84, 4]  # Our input is a stack of 4 frames hence 84x84x4 (Width, height, channels)
        action_size = 4
        learning_rate = 0.0002  # Alpha (aka learning rate)
        # Reset the graph
        tf.reset_default_graph()

        # Instantiate the DQNetwork
        self.net = DQNetwork(state_size, action_size, learning_rate)
        super().__init__(width, height, local_rewards)

    def get_action(self, eps):
        number = np.random.uniform()
        action = None
        if number <= eps:
            # explore
            action = random.choice(list(self.Q[self.current_state].keys()))
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

    def pretrain(self, memory):
        mw.current_state = (0, 0)
        for i in range(pretrain_length):
            # If it's the first step
            if i == 0:
                # First we need a state
                state = self.current_state

            # Random action
            action = self.get_action(1)

            # Get the rewards
            reward = self.make_action(action)

            # Look if the episode is finished
            done = self.game_over()

            # If we're dead
            if done:
                # We finished the episode
                next_state = (-1, -1)

                # Add experience to memory
                memory.add((state, action, reward, next_state, done))

                # Start a new episode
                mw.current_state = (0, 0)

                # First we need a state
                state = self.current_state
            else:
                # Get the next state
                next_state = self.current_state

                # Add experience to memory
                memory.add((state, action, reward, next_state, done))

                # Our state is now the next_state
                state = next_state

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

    mw.pretrain(memory)
    with tf.Session() as sess:
        # Initialize the variables
        sess.run(tf.global_variables_initializer())
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
                    memory.add((state, action, reward, (-1, -1), False))
                else:
                    memory.add((state, action, reward, next_state, False))

            batch = memory.sample(batch_size)
            states_mb = np.array([each[0] for each in batch], ndmin=3)
            actions_mb = np.array([each[1] for each in batch])
            rewards_mb = np.array([each[2] for each in batch])
            next_states_mb = np.array([each[3] for each in batch], ndmin=3)
            dones_mb = np.array([each[4] for each in batch])

            # Get Q values for next_state
            Qs_next_state = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: next_states_mb})

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
            sess.run([DQNetwork.loss, DQNetwork.optimizer],
                     feed_dict={DQNetwork.inputs_: states_mb,
                                DQNetwork.target_Q: targets_mb,
                                DQNetwork.actions_: actions_mb})

    end = time.time()
    mw.print_best_policy()
    print('Finished')
    print('time: {}'.format(end - start))