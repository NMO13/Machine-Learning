import numpy as np
import random
import collections

class Game:
    ROWS = 3
    COLUMNS = 3
    TILES_TO_WIN = 3
    RED = 1
    YELLOW = -1
    GAMMA = 0.9

    def __init__(self):
        self.board = np.zeros( (Game.ROWS, Game.COLUMNS) )
        self.actions = np.arange(0, Game.COLUMNS)

    def reset(self, p1, p2):
        self.board = np.zeros((Game.ROWS, Game.COLUMNS))
        p1.states_actions_rewards = []
        p2.states_actions_rewards = []


    def _other_player(self, cur_player, p1, p2):
        if cur_player == p1:
            return p2
        return p1

    def play(self, p1, p2):
        self.reset(p1, p2)
        cur_player = p1
        game_over = False

        prev_state = 0
        while not game_over:
            reward, action, game_over, valid_move = cur_player.move(epsilon, self)
            if valid_move:
                cur_player.states_actions_rewards.append((prev_state, action, reward))
                self._other_player(cur_player, p1, p2).states_actions_rewards.append((prev_state, action, -reward))
            else:
                cur_player.states_actions_rewards.append((prev_state, action, reward))
                self._other_player(cur_player, p1, p2).states_actions_rewards.append((prev_state, action, reward))
            cur_player = p2 if cur_player == p1 else p1
            prev_state = self.get_state()


        return self.create_sar(p1.states_actions_rewards), self.create_sar(p2.states_actions_rewards)

    def create_sar(self, states_actions_rewards):
        sar = []
        G = 0
        for s, a, r in reversed(states_actions_rewards):
            G = r + Game.GAMMA * G
            sar.append((s, a, G))

        sar.reverse()
        return sar

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
        if not 0 in self.board:
            return True, 0

        row = self.board[self.latest_move[0]]
        if self._check_adjacency(row, sym):
            return True, 1
        if self._check_adjacency(row, -sym):
            return True, -1

        col = self.board[:,self.latest_move[1]]
        if self._check_adjacency(col, sym):
            return True, 1
        if self._check_adjacency(col, -sym):
            return True, -1

        diag = self.get_diagonal(self.latest_move)
        if self._check_adjacency(diag, sym):
            return True, 1
        if self._check_adjacency(diag, -sym):
            return True, -1

        diag = self.get_diagonal(self.latest_move, True)
        if self._check_adjacency(diag, sym):
            return True, 1
        if self._check_adjacency(diag, -sym):
            return True, -1

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
        return h

    def get_valid_actions(self):
        valid_actions = []
        for i in range(Game.COLUMNS):
            if not self.board[0, i]:
                valid_actions.append(i)
        return valid_actions

class Player:
    def __init__(self, sym):
        self.sym = sym
        self.Q = {}
        self.states_actions_rewards = []
        self.policy = {}
        self.return_dict = {}

    def move(self, eps, env):
        number = np.random.uniform()
        action = None
        if number <= eps:
            # explore
            valid_actions = env.get_valid_actions()
            action = random.choice(valid_actions)
        else:
            # exploit
            action = self.policy.get(env.get_state(), None)
            if action == None:
                valid_actions = env.get_valid_actions()
                action = random.choice(valid_actions)

        if env.board[0, action]:
            return -100, action, True, False
        env.set_sym(action, self.sym)
        game_over, reward = env.game_over(self.sym)
        return reward, action, game_over, True

    def update_q_and_policy(self, sar):
        for state, action, G in sar:
            sa = (state, action)
            self.return_dict.setdefault(sa, []).append(G)
            aval = self.Q.setdefault(state, {})
            aval.setdefault(action, 0)
            aval[action] = np.mean(self.return_dict[sa])
            action = max(self.Q[state].keys(), key=(lambda key: self.Q[state][key]))
            self.policy[state] = action

if __name__ == "__main__":
    import time
    start = time.time()
    mw = Game()
    # Exploration parameters
    epsilon = 0.2  # Exploration rate
    max_epsilon = 1.0  # Exploration probability at start
    min_epsilon = 0.01  # Minimum exploration probability
    decay_rate = 0.005  # Exponential decay rate for exploration prob

    p1 = Player(Game.RED)
    p2 = Player(Game.YELLOW)

    for episode in range(10000):
        if episode % 100 == 0:
            print(episode)
        sar1, sar2 = mw.play(p1, p2)
        p1.update_q_and_policy(sar1)
        p2.update_q_and_policy(sar2)

        # Reduce epsilon (because we need less and less exploration)
        #epsilon = 1/(episode+1)#min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    end = time.time()
    print('Finished')
    print('time: {}'.format(end - start))


