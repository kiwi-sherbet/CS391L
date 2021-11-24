#! /usr/bin/python3

import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import copy

PATH_SCRIPT = os.path.dirname(os.path.realpath(__file__))
PATH_ROOT   = os.path.dirname(PATH_SCRIPT)
PATH_SAVE   = os.path.join(PATH_ROOT, 'save')
PATH_PLOT   = os.path.join(PATH_ROOT, 'plot')

EPSILON = 0.25
GAMMA = 0.90
ALPHA = 0.05

FULL_GRID_X = 6
FULL_GRID_Y = 25
LITTER_GRID_DIM = 5
OBSTACLES_GRID_DIM = 3
LITTER_PROB = 1 / 20
OBSTACLES_PROB = 1 / 5

EPOCH_ITERATIONS = 5000

WEIGHTS = {'litter': 0.4, 'obstacle' : 0.2, 'sidewalk': 0.2, 'forward': 0.2}
# WEIGHTS = {'litter': 0.1, 'obstacle' : 0.3, 'sidewalk': 0.3, 'forward': 0.3}
# Uncomment below weights for testing single-task module
# WEIGHTS = {'litter': 0.5, 'obstacle' : 0.0, 'sidewalk': 0.0, 'forward': 0.5}
# WEIGHTS = {'litter': 0.0, 'obstacle' : 0.5, 'sidewalk': 0.0, 'forward': 0.5}
# WEIGHTS = {'litter': 0.0, 'obstacle' : 0.0, 'sidewalk': 0.5, 'forward': 0.5}
# WEIGHTS = {'litter': 0.0, 'obstacle' : 0.0, 'sidewalk': 0.0, 'forward': 1.0}

COMBINED_EPSILON = 0.1

NUM_ACTIONS = 8
ACTION_MAP = {  0 : np.array((0,1)),
                1 : np.array((1,1)), 
                2 : np.array((1,0)), 
                3 : np.array((1,-1)), 
                4 : np.array((0,-1)), 
                5 : np.array((-1,-1)), 
                6 : np.array((-1,0)),
                7 : np.array((-1,1))}

SIDEWALK_ROWS = set([1, 2, 3, 4])

os.makedirs(PATH_DATA, exist_ok=True)
os.makedirs(PATH_SAVE, exist_ok=True)
os.makedirs(PATH_PLOT, exist_ok=True)

def ndarray_to_tuple(array):
    return tuple(np.ravel(array).tolist())


def tuple_to_ndarray(tup, d):
    l = [*tup]
    return np.reshape(np.array(l), (d, d))


def get_action(state, q_table, eps):
    opt = get_argmax(q_table, state)
    dist = eps / NUM_ACTIONS * np.ones((NUM_ACTIONS,))
    dist[opt] += 1 - eps
    return np.random.choice(NUM_ACTIONS, p=dist)


def get_q_val(q_table, key):
    if key in q_table.keys():
        q_val = q_table[key]
    else:
        q_val = 0
    return q_val


def get_argmax(q_table, state):
    maxes = []
    max_q = -1 * np.inf
    for i in ACTION_MAP.keys():
        key = (ndarray_to_tuple(state), i)
        q_val = get_q_val(q_table, key)
        if q_val > max_q:
            maxes = [i]
            max_q = q_val
        elif q_val == max_q:
            maxes.append(i)
    opt = np.random.choice(maxes)
    return opt


def rescale_q_val(q_val, max_q, min_q):
    if max_q == min_q:
        return 0.5
    return (q_val - min_q) / (max_q - min_q)


def epsilon_greedy(opt, eps):
    
    dist = eps / NUM_ACTIONS * np.ones(NUM_ACTIONS)
    dist[opt] += 1 - eps
    return np.random.choice(NUM_ACTIONS, p=dist)


def litter_init(grid_shape = np.array([LITTER_GRID_DIM, LITTER_GRID_DIM]),
                pos_init = np.array([LITTER_GRID_DIM//2, LITTER_GRID_DIM//2])
                ):
    p = LITTER_PROB
    grid = np.zeros(grid_shape)
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            pos = np.array([i,j])
            r = np.random.rand()
            if np.array_equal(pos, pos_init):
                continue
            elif r < p:
                grid[pos[0], pos[1]] = 1
    return grid


def litter_step(grid, action,
                grid_shape = np.array([LITTER_GRID_DIM, LITTER_GRID_DIM]),
                pos_cur = np.array([LITTER_GRID_DIM//2, LITTER_GRID_DIM//2])
                ):
    grid_new = np.zeros(grid_shape)
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            pos = np.array([i,j])
            pos_new = pos + action
            if np.array_equal(pos, pos_cur):
                # Pick up litter case
                grid_new[pos[0], pos[1]] = 0
            elif np.array_equal(pos_new, np.clip(pos_new, 0, grid_shape - 1)):
                grid_new[pos[0], pos[1]] = grid[pos_new[0], pos_new[1]]
            else:
                grid_new[pos[0], pos[1]] = 0
    return grid_new


def litter_reward(state, action, new_state,
                 grid_shape = np.array([LITTER_GRID_DIM, LITTER_GRID_DIM]),
                 pos_cur = np.array([LITTER_GRID_DIM//2, LITTER_GRID_DIM//2])
                 ):
    pos_new = pos_cur + action
    if state[pos_new[0], pos_new[1]] == 1:
        return 1
    else:
        return 0


def litter_done(state,
                grid_shape = np.array([LITTER_GRID_DIM, LITTER_GRID_DIM]),
                ):
    terminal_state = np.zeros(grid_shape)
    if np.array_equal(state, terminal_state):
        return True
    return False


def obstacle_init(grid_shape = np.array([OBSTACLES_GRID_DIM, OBSTACLES_GRID_DIM]),
                  pos_init = np.array([OBSTACLES_GRID_DIM//2, OBSTACLES_GRID_DIM//2])
                  ):
    p = OBSTACLES_PROB
    grid = np.zeros(grid_shape)
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            pos = np.array([i,j])
            r = np.random.rand()
            if np.array_equal(pos, pos_init):
                continue        
            elif r < p:
                grid[pos[0], pos[1]] = 1
    return grid


def obstacle_step(grid, action,
                  grid_shape = np.array([OBSTACLES_GRID_DIM, OBSTACLES_GRID_DIM]),
                  pos_cur = np.array([OBSTACLES_GRID_DIM//2, OBSTACLES_GRID_DIM//2])
                  ):
    grid_new = np.zeros(grid_shape)
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            pos = np.array([i,j])
            pos_new = pos + action
            if np.array_equal(pos_new, np.clip(pos_new, 0, grid_shape - 1)):
                grid_new[pos[0], pos[1]] = grid[pos_new[0], pos_new[1]]
            else:
                grid_new[pos[0], pos[1]] = 0
    return grid_new


def obstacle_reward(state, action, new_state,
                    grid_shape = np.array([OBSTACLES_GRID_DIM, OBSTACLES_GRID_DIM]),
                    pos_cur = np.array([OBSTACLES_GRID_DIM//2, OBSTACLES_GRID_DIM//2])
                    ):
    pos_new = pos_cur + action
    if state[pos_new[0], pos_new[1]] == 1:
        return -1
    else:
        return 0


def obstacle_done(state,
                  grid_shape = np.array([OBSTACLES_GRID_DIM, OBSTACLES_GRID_DIM]),
                  ):
    terminal_state = np.zeros(grid_shape)
    if np.array_equal(state, terminal_state):
        return True
    return False


def walk_init(grid_shape = np.array([FULL_GRID_X, FULL_GRID_Y]),
              pos_init = None):
    
    if pos_init is None:
        r = np.random.choice(4) + 1
        return np.array([r, 0])

    else:
        return pos_init


def walk_step(pos_cur, action,
              grid_shape = np.array([FULL_GRID_X, FULL_GRID_Y]),
              ):
    pos_new = pos_cur + action
    return np.clip(pos_new, np.zeros(2), grid_shape -1)


def forward_reward(pos_cur, action, pos_new,
                   grid_shape = np.array([FULL_GRID_X, FULL_GRID_Y]),
                   ):
    if pos_new[1] == grid_shape[1] - 1:
        return 1
    else:
        return 0


def sidewalk_reward(pos_cur, action, new_state,
                   grid_shape = np.array([FULL_GRID_X, FULL_GRID_Y]),
                   ):
    if new_state[0] in SIDEWALK_ROWS:
        return 0
    else:
        return -1


def walk_done(state,
              grid_shape = np.array([FULL_GRID_X, FULL_GRID_Y]),
              ):
    if state[1] == grid_shape[1] - 1:
        return True
    else:
        return False
        

class Episode():
    def __init__(self, modules) -> None:
        self.modules = modules
        self.grid_shape = np.array([FULL_GRID_X, FULL_GRID_Y])
        self.litter_margin = LITTER_GRID_DIM // 2
        self.obstacle_margin = OBSTACLES_GRID_DIM // 2
        self.history = []

    def reset(self):
        self.pos = self.modules['forward'].init(grid_shape = self.grid_shape, pos_init = None)
        self.litter_grid = self.modules['litter'].init(grid_shape = self.grid_shape, pos_init = self.pos)
        self.obstacles_grid = self.modules['obstacle'].init(grid_shape = self.grid_shape, pos_init = self.pos)

        for i in range(self.grid_shape[0]):
            for j in range(self.grid_shape[1]):
                if self.obstacles_grid[i, j]:
                    self.litter_grid[i, j] = 0

        self.litter_grid_ext = np.zeros(episode.grid_shape + 2 * self.litter_margin)
        self.litter_grid_ext[self.litter_margin:self.litter_margin + self.grid_shape[0], 
                             self.litter_margin:self.litter_margin + self.grid_shape[1]] = self.litter_grid
        self.obstacle_grid_ext = np.zeros(episode.grid_shape + 2 * self.obstacle_margin)
        self.obstacle_grid_ext[self.obstacle_margin:self.obstacle_margin + self.grid_shape[0], 
                               self.obstacle_margin:self.obstacle_margin + self.grid_shape[1]] = self.obstacles_grid

        self.litter_grid_local = self.litter_grid_ext[self.pos[0]: self.pos[0] + 2 * self.litter_margin + 1, 
                                                      self.pos[1]: self.pos[1] + 2 * self.litter_margin + 1]
        self.obstacle_grid_local = self.obstacle_grid_ext[self.pos[0]: self.pos[0] + 2 * self.obstacle_margin + 1,
                                                          self.pos[1]: self.pos[1] + 2 * self.obstacle_margin + 1]
        
        self.litter_grid_init = copy.copy(self.litter_grid)
        self.history.append(copy.copy(self.pos))


    def seed(self, seed = 0):
        np.random.seed(seed)


    def step(self, action):
        print(self.pos)
        self.pos += action
        self.pos = np.clip(self.pos, 0, self.grid_shape - 1)
        self.litter_grid[self.pos[0], self.pos[1]] = 0

        self.litter_grid_ext[self.litter_margin:self.litter_margin + self.grid_shape[0], 
                             self.litter_margin:self.litter_margin + self.grid_shape[1]] = self.litter_grid

        self.litter_grid_local = self.litter_grid_ext[self.pos[0]: self.pos[0] + 2 * self.litter_margin + 1, 
                                                      self.pos[1]: self.pos[1] + 2 * self.litter_margin + 1]
        self.obstacle_grid_local = self.obstacle_grid_ext[self.pos[0]: self.pos[0] + 2 * self.obstacle_margin + 1,
                                                          self.pos[1]: self.pos[1] + 2 * self.obstacle_margin + 1]
        self.history.append(copy.copy(self.pos))

    def done(self):
        return self.modules['forward'].done(self.pos)


    def get_pos(self):
        return copy.copy(self.pos)


    def get_litter_grid(self):
        return copy.copy(self.litter_grid_local)


    def get_obstacle_grid(self):
        return copy.copy(self.obstacle_grid_local)


    def plot_state(self):
        plt.figure()
        litter = []
        obstacle = []
        for i in range(self.grid_shape[0]):
            for j in range(self.grid_shape[1]):
                pos = np.array([i,j])
                if self.litter_grid_init[pos[0], pos[1]] == 1:
                    litter.append(pos)
                if self.obstacles_grid[pos[0], pos[1]] == 1:
                    obstacle.append(pos)


        litter = np.array(litter)
        obstacle = np.array(obstacle)
        history = np.array(self.history)

        # Uncomnnemt the scatters for visualizing single-task modules
        plt.scatter(litter[:, 0], litter[:, 1], c='g', marker='o', label='Litter')
        plt.scatter(obstacle[:, 0], obstacle[:, 1], c='r', marker='X', label='Obstacle')
        plt.scatter(np.concatenate((np.zeros(25), 5*np.ones(25)), axis=0),
                    np.concatenate((np.arange(25), np.arange(25)), axis=0),
                    c='orange', marker="4", label='Out of Sidewalk')
        plt.plot(history[:, 0], history[:, 1], c='k', label="Trajectory")
        plt.ylabel('Y coordinate (forward direction)')
        plt.xlabel('X coordinate (lateral direction)')
        plt.legend(loc='upper right')
        plt.savefig('{}/eval.png'.format(PATH_PLOT))
        plt.show()
        plt.close()


class Module():
    def __init__(self, init, step, reward, done) -> None:        
        self.init = init
        self.step = step
        self.reward = reward
        self.done = done

    def train(self, q_table):
        state = self.init()
        delta = 0
        for _ in range(EPOCH_ITERATIONS):
            if self.done(state):
                state = self.init()
            action_key = get_action(state, q_table, EPSILON)
            action = ACTION_MAP[action_key]
            key = (ndarray_to_tuple(state), action_key)
            new_state = self.step(state, action)
            reward = self.reward(state, action, new_state)
            q_argmax = get_q_val(q_table, (ndarray_to_tuple(new_state), get_argmax(q_table, new_state)))
            q_update = reward + GAMMA * q_argmax
            q_prev = get_q_val(q_table, key)
            q_cur = (1 - ALPHA) * q_prev + ALPHA * q_update
            delta += np.abs(q_prev - q_cur)
            q_table[key] = q_cur
            state = new_state
        return q_table, delta / EPOCH_ITERATIONS


if __name__=="__main__":
    modules = {}
    q_tables = {}

    modules['litter'] = Module(litter_init, litter_step, litter_reward, litter_done)
    modules['obstacle'] = Module(obstacle_init, obstacle_step, obstacle_reward, obstacle_done)
    modules['sidewalk'] = Module(walk_init, walk_step, sidewalk_reward, walk_done)
    modules['forward'] = Module(walk_init, walk_step, forward_reward, walk_done)

    # for task, module in modules.items():

    #     q_table = {}
    #     avg_deltas = []
    #     episodes = []


    #     for i in range(1500):
    #         # The training for litter collection
    #         q_table, avg_delta = module.train(q_table)
    #         avg_deltas.append(avg_delta)
    #         episodes.append(i + 1)
    #         print("********** Epoch #{} **********".format((i + 1)))
    #         print("Q table entries: {}".format(len(q_table.keys())))
    #         pair = q_table.popitem()
    #         q_table[pair[0]] = pair[1]
    #         state = pair[0][0]

    #         if task == 'litter':
    #             print("State:\n{}".format(tuple_to_ndarray(state, LITTER_GRID_DIM)))
    #             opt = get_argmax(q_table, state)
    #             print("Optimal Action: {}".format(ACTION_MAP[opt]))
    #             print("Q value: {}".format(get_q_val(q_table, (ndarray_to_tuple(state), opt))))
    #         elif task == 'obstacle':
    #             print("State:\n{}".format(tuple_to_ndarray(state, OBSTACLES_GRID_DIM)))
    #             opt = get_argmax(q_table, state)
    #             print("Optimal Action: {}".format(ACTION_MAP[opt]))
    #             print("Q value: {}".format(get_q_val(q_table, (ndarray_to_tuple(state), opt))))
    #         else:
    #             print("State: {}".format(state))

    #         with open('{}/{}.pickle'.format(PATH_SAVE, task), 'wb') as handle:
    #             pickle.dump(q_table, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #         plt.figure(0)
    #         plt.plot(episodes, avg_deltas)
    #         plt.ylabel('Q value differene')
    #         plt.xlabel('Epoch')
    #         plt.savefig('{}/{}.png'.format(PATH_PLOT, task))
    #         plt.close()


    for task, module in modules.items():

        with open('{}/{}.pickle'.format(PATH_SAVE, task), 'rb') as handle:
            q_tables[task] = pickle.load(handle)


    episode = Episode(modules)
    episode.seed()
    episode.reset()

    while not episode.done():
        
        q_val_mins = {}
        q_val_maxes = {}
        states = {}

        for task, module in modules.items():
            q_val_mins[task] = np.inf
            q_val_maxes[task] = -np.inf

        states['litter'] = ndarray_to_tuple(episode.get_litter_grid())
        states['obstacle'] = ndarray_to_tuple(episode.get_obstacle_grid())
        states['forward'] = ndarray_to_tuple(episode.get_pos())
        states['sidewalk'] = ndarray_to_tuple(episode.get_pos())
 
        for action_key in ACTION_MAP.keys():

            for task, module in modules.items():

                q_val = get_q_val(q_tables[task], (states[task], action_key))

                if q_val < q_val_mins[task]:
                    q_val_mins[task] = q_val

                if q_val > q_val_maxes[task]:
                    q_val_maxes[task] = q_val

        maxes = []
        max_s = -1 * np.inf

        for action_key in ACTION_MAP.keys():

            vals = {}
            s = 0

            for task, module in modules.items():

                vals[task] = rescale_q_val(get_q_val(q_tables[task], (states[task], action_key)), q_val_maxes[task], q_val_mins[task])
                s += WEIGHTS[task] * vals[task]

            action = ACTION_MAP[action_key]
            print("{} weighted value: {}".format(action, s))
            if s > max_s:
                maxes = [action_key]
                max_s = s
            elif s == max_s:
                maxes.append(action_key)

        opt = np.random.choice(maxes)
        action_key = epsilon_greedy(opt, COMBINED_EPSILON)
        action = ACTION_MAP[action_key]
        print("Selected action: {}".format(action))
        episode.step(action)
    episode.plot_state()

