from lib import wrappers
from lib import dqn_model

import argparse
import time
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter

import gym

MEAN_REWARD_BOUND = 19.5

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 100
REPLAY_START_SIZE = 500

EPSILON_DECAY_LAST_FRAME = 10**5
EPSILON_START = 1.0
EPSILON_FINAL = 0.02

HIDDEN_SIZE = 128

ENV = 'connect4'
class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size[0], hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.double), \
               np.array(dones, dtype=np.uint8), np.array(next_states)


class Agent:
    def __init__(self, env, exp_buffer, sym, net):
        self.env = env
        self.exp_buffer = exp_buffer
        self.sym = sym
        self.net = net
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    def play_step(self, other_agent, epsilon=0.0, device="cpu"):
        done_reward = None

        valid_actions = self.env.get_valid_actions()
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
            while not action in valid_actions:
                action = self.env.action_space.sample()

        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            #res = self.net.double()
            #q_vals_v2 = res(state_v.view(9))
            valid_actions_v = torch.tensor(valid_actions).to(device)
            q_vals_v = self.net(state_v.view(9).float())
            valid_q_vals_v = q_vals_v.gather(0, valid_actions_v)
            x, act_v = torch.max(valid_q_vals_v, dim=0)
            index = (q_vals_v == x).nonzero()
            action = int(index)

        # do step in the environment
        new_state, reward, is_done = self.env.step(action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)

        exp_other = Experience(self.state, action, 0, is_done, new_state)
        other_agent.exp_buffer.append(exp_other)

        self.state = new_state

        other_agent.state = new_state

        if is_done:
            done_reward = self.total_reward
            self._reset()
            other_agent._reset()
        return done_reward, is_done

class HumanPlayer(Agent):
    def __init__(self, env, sym):
        super(HumanPlayer, self).__init__(env, ExperienceBuffer(REPLAY_SIZE), sym, None)

    epsilon = 0
    def play_step(self, other_agent, epsilon=0.0, device="cpu"):
        valid_actions = env.get_valid_actions()
        while True:
            # break if we make a legal move
            move = input("Enter column: ")
            if int(move) in valid_actions:
                env.set_sym(int(move), self.sym)
                break
        game_over, reward = env.game_over(self.sym)
        return reward, game_over

class Game(gym.Env):
    ROWS = 3
    COLUMNS = 3
    TILES_TO_WIN = 3
    RED = 1
    YELLOW = -1
    GAMMA = 0.9

    def __init__(self):
        self.board = np.zeros( (Game.ROWS, Game.COLUMNS) )
        self.actions = np.arange(0, Game.COLUMNS)
        self.action_space = gym.spaces.Discrete(n=Game.COLUMNS)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(Game.ROWS * Game.COLUMNS, ), dtype=np.double)

    def draw_board(self):
        for i in range(Game.ROWS):
            print('----' * Game.COLUMNS)
            for j in range(Game.COLUMNS):
                print("  ", end="")
                if self.board[i, j] == Game.RED:
                    print("x ", end="")
                elif self.board[i, j] == Game.YELLOW:
                    print("o ", end="")
                else:
                    print("  ", end="")
            print("")
        print('----' * Game.COLUMNS)

    def reset(self):
        self.board = np.zeros((Game.ROWS, Game.COLUMNS), dtype=np.double)
        return np.copy(self.board)


    def _other_player(self, cur_player, p1, p2):
        if cur_player == p1:
            return p2
        return p1

    def get_cur_player(self):
        r_counter = 0
        y_counter = 0
        for i in range(Game.ROWS):
            for j in range(Game.COLUMNS):
                if self.board[i][j] == Game.RED:
                    r_counter += 1
                elif self.board[i][j] == Game.YELLOW:
                    y_counter += 1

        if r_counter > y_counter:
            return p2
        elif r_counter < y_counter:
            return p1

        return p1

    def step(self, action):
        player = self.get_cur_player()
        env.set_sym(action, player.sym)
        game_over, reward = env.game_over(player.sym)
        return np.copy(env.board), reward, game_over

    def set_sym(self, col, sym):
        col_vals = self.board[:,col]
        for idx, val in enumerate(col_vals[::-1]):
            if not val:
                self.board[Game.ROWS - idx - 1, col] = sym
                self.latest_move = (Game.ROWS - idx - 1, col)
                break

    def _check_adjacency(self, array, sym):
        adjacent_counter = 0
        for element in array:
            if element != sym:
                adjacent_counter = 0
            else:
                adjacent_counter += 1
            if adjacent_counter == Game.TILES_TO_WIN:
                return True
        return False

    def game_over(self, sym):
        reward = 10

        row = self.board[self.latest_move[0]]
        if self._check_adjacency(row, sym):
            return True, reward
        if self._check_adjacency(row, -sym):
            return True, 0

        col = self.board[:,self.latest_move[1]]
        if self._check_adjacency(col, sym):
            return True, reward
        if self._check_adjacency(col, -sym):
            return True, 0

        diag = self.get_diagonal(self.latest_move)
        if self._check_adjacency(diag, sym):
            return True, reward
        if self._check_adjacency(diag, -sym):
            return True, 0

        diag = self.get_diagonal(self.latest_move, True)
        if self._check_adjacency(diag, sym):
            return True, reward
        if self._check_adjacency(diag, -sym):
            return True, 0

        if not 0 in self.board:
            return True, 0

        return False, 0

    def get_diagonal(self, pos, reversed=False):
        if reversed:
            self.board = self.board[:, ::-1]
            new_y = Game.COLUMNS - 1 - pos[1]
            pos = (pos[0], new_y)
        def in_range(x, y):
            if x < 0 or y < 0: return False
            if x >= self.board.shape[0] or y >= self.board.shape[1]:
                return False
            return True

        up_values = []
        x = pos[0]
        y = pos[1]
        while True:
            if not in_range(x, y):
                break
            up_values.append(self.board[x, y])
            x -= 1
            y -= 1
        up_values.reverse()

        x = pos[0] + 1
        y = pos[1] + 1
        while True:
            if not in_range(x, y):
                break
            up_values.append(self.board[x, y])
            x += 1
            y += 1

        if reversed:
            self.board = self.board[:, ::-1]
        return up_values


    def get_state(self):
        # returns the current state, represented as an int
        # from 0...|S|-1, where S = set of all possible states
        # |S| = 3^(BOARD SIZE), since each cell can have 3 possible values - empty, x, o
        # some states are not possible, e.g. all cells are x, but we ignore that detail
        # this is like finding the integer represented by a base-3 number
        k = 0
        h = 0
        for i in range(self.COLUMNS):
            for j in range(self.ROWS):
                if self.board[i, j] == 0:
                    v = 0
                elif self.board[i, j] == self.RED:
                    v = 1
                elif self.board[i, j] == self.YELLOW:
                    v = 2
                h += (3 ** k) * v
                k += 1
        board_copy = np.copy(self.board)
        self.set_board(h)
        assert np.array_equal(board_copy.all(), self.board.all())
        return h

    def set_board(self, state):
        import math
        vals = np.repeat(0, self.COLUMNS * self.ROWS)
        i = 0
        while state > 0:
            vals[i] = state % 3
            state = math.floor(state / 3)
            i = i + 1
        for i in range(self.ROWS):
            for j in range(self.COLUMNS):
                val =  vals[j + i * Game.ROWS]
                if val == 0:
                    self.board[i][j] = 0
                elif val == 1:
                    self.board[i][j] = self.RED
                elif val == 2:
                    self.board[i][j] = self.YELLOW

    def get_valid_actions(self):
        valid_actions = []
        for i in range(Game.COLUMNS):
            if not self.board[0, i]:
                valid_actions.append(i)
        return valid_actions

def calc_loss(batch, net, tgt_net, device="cpu"):
    states, actions, rewards, dones, next_states = batch

    states_v = torch.tensor(states).float().to(device)
    next_states_v = torch.tensor(next_states).float().to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).float().to(device)
    done_mask = torch.ByteTensor(dones).to(device)

    state_action_values = net(states_v.view((-1, 9))).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_values = tgt_net(next_states_v.view((-1, 9))).max(1)[0]
    next_state_values[done_mask] = 0.0
    next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * GAMMA + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)

if __name__ == "__main__":
    # Exploration parameters
    max_epsilon = 1  # Exploration probability at start
    min_epsilon = 0.2  # Minimum exploration probability
    decay_rate = 2000  # Exponential decay rate for exploration prob

    debug = False
    import time
    start = time.time()
    env = Game()

    device = torch.device("cuda")

    net1 = Net(env.observation_space.shape, HIDDEN_SIZE, env.action_space.n).to(device)
    tgt_net1 = Net(env.observation_space.shape, HIDDEN_SIZE, env.action_space.n).to(device)
    optimizer1 = optim.Adam(net1.parameters(), lr=LEARNING_RATE)

    net2 = Net(env.observation_space.shape, HIDDEN_SIZE, env.action_space.n).to(device)
    tgt_net2 = Net(env.observation_space.shape, HIDDEN_SIZE, env.action_space.n).to(device)
    optimizer2 = optim.Adam(net2.parameters(), lr=LEARNING_RATE)


    writer = SummaryWriter(comment="-" + ENV)

    buffer1 = ExperienceBuffer(REPLAY_SIZE)
    buffer2 = ExperienceBuffer(REPLAY_SIZE)
    p1 = Agent(env, buffer1, Game.RED, net1)
    p2 = Agent(env, buffer2, Game.YELLOW, net2)
    epsilon = EPSILON_START

    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_mean_reward = None
    current_player = p1

    while True:
        frame_idx += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)

        if current_player is p1:
            done, reward = p1.play_step(p2, epsilon, device=device)
            current_player = p2
        else:
            done, reward = p2.play_step(p1, epsilon, device=device)
            current_player = p1

        if done and current_player is p2:
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            mean_reward = np.mean(total_rewards[-100:])
            print("%d: done %d games, mean reward %.3f, eps %.2f, speed %.2f f/s" % (
                frame_idx, len(total_rewards), mean_reward, epsilon,
                speed
            ))
            print("Buffer size: %d" % len(buffer1))
            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", mean_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)
            if best_mean_reward is None or best_mean_reward < mean_reward:
                torch.save(net1.state_dict(), ENV + "-best.dat")
                if best_mean_reward is not None:
                    print("Best mean reward updated %.3f -> %.3f, model saved" % (best_mean_reward, mean_reward))
                best_mean_reward = mean_reward
            current_player = p1
            if len(total_rewards) > 1000:
                print("Solved in %d frames!" % frame_idx)
                break

        assert(len(buffer1) == len(buffer2))
        if len(buffer1) < REPLAY_START_SIZE:
            continue

        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net1.load_state_dict(net1.state_dict())
            tgt_net2.load_state_dict(net2.state_dict())

        optimizer1.zero_grad()
        loss_t1 = calc_loss(buffer1.sample(BATCH_SIZE), net1, tgt_net1, device=device)
        loss_t1.backward()
        optimizer1.step()
        writer.add_scalar("loss1", loss_t1.item(), frame_idx)

        optimizer2.zero_grad()
        loss_t2 = calc_loss(buffer2.sample(BATCH_SIZE), net2, tgt_net2, device=device)
        loss_t2.backward()
        optimizer2.step()

        writer.add_scalar("loss2", loss_t2, frame_idx)

    writer.close()

    env = Game()
    p1.env = env
    p2 = HumanPlayer(env, Game.YELLOW)
    epsilon = 0

    current_player = p1
    debug = True
    while True:
        done = False
        while not done:
            if current_player is p1:
                reward, done = p1.play_step(p2, 0, device=device)
                current_player = p2
                env.draw_board()
            else:
                reward, done = p2.play_step(p1, 0, device=device)
                current_player = p1

        current_player = p1
        env.reset()

        answer = input("Play again? [Y/n]: ")
        if answer and answer.lower()[0] == 'n':
            break