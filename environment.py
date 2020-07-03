import numpy as np
import pandas as pd
import random
from scipy.stats import binom


class GridWorld(object):
    def __init__(self, shape, actions, starts=None, terminal=None, walls=None, inf_actions=None):
        self.n_rows, self.n_columns = shape
        self.grid_size = self.n_rows * self.n_columns

        self.starts = []
        if starts is not None:
            self.starts = [self.xy2s(pos[0], pos[1]) for pos in starts]

        self.terminal = []
        extra_terminal = 0
        n_terminal_states = 0
        if terminal is not None:
            n_terminal_states = 1
            self.terminal = [pos[0] * self.n_columns + pos[1]
                             for pos in terminal]
            if self.terminal[0] >= self.n_rows * self.n_columns:
                extra_terminal = 1

        self.walls = []
        if walls is not None:
            self.walls = [self.xy2s(pos[0], pos[1]) for pos in walls]

        self.actions = actions
        self.n_actions = len(actions)

        self.n_states = self.n_rows * self.n_columns + extra_terminal
        self.n_nonterminal_states = self.n_states - n_terminal_states

        self.action_sets = None
        if inf_actions is not None:
            action_sets = dict()
            for state, inf_action in inf_actions:
                action_sets[self.xy2s(state[0], state[1])] = [[i for i, ai in enumerate(
                    actions) if ai not in inf_action], [i for i, ai in enumerate(actions) if ai in inf_action]]
            self.action_sets = action_sets

        states = []
        for x in range(self.n_rows):
            rows = []
            for y in range(self.n_columns):
                rows.append((x, y))
            states.append(rows)
        self.states = states

        self.T = np.zeros((self.n_states, self.n_actions, self.n_states))
        self.R = np.zeros((self.n_states, self.n_actions))

    def check_model(self):
        for s in range(self.n_states):
            if s not in self.terminal and s not in self.walls:
                action_sets = range(self.n_actions)
                if self.action_sets is not None and s in self.action_sets:
                    action_sets = self.action_sets[s][0]

                for a in action_sets:
                    if abs(sum(self.T[s, a, :]) - 1) >= 1e-10:
                        raise Exception('The transition probability for state %d and action %s should sum as 1' % (
                            s, self.actions[a]))

        return True

    def xy2s(self, x, y):
        x = max(x, 0)
        y = max(y, 0)
        x = min(x, self.n_rows - 1)
        y = min(y, self.n_columns - 1)

        return x * self.n_columns + y

    def xy2s_wall(self, x, y, s0):
        s = self.xy2s(x, y)
        if s in self.walls:
            return s0
        else:
            return s

    def s2xy(self, s):
        return s // self.n_columns, s % self.n_columns

    def s2tile(self, s):
        return (s // self.n_columns) / (self.n_rows - 1), (s % self.n_columns) / (self.n_columns - 1)

    def step(self, s, a):
        p = self.T[s, a, :]
        next_state = random.choices(range(self.n_states), weights=p, k=1)[0]
        return next_state, self.R[s, a], self.is_terminal(next_state)

    def initialize_state(self):
        if len(self.starts) != 0:
            return random.choices(self.starts, k=1)[0]
        else:
            s = random.choices(range(self.grid_size), k=1)[0]
            while s in self.walls or s in self.terminal:
                s = random.choices(range(self.grid_size), k=1)[0]

            return s

    def is_terminal(self, s):
        return s in self.terminal

    def show_state_value(self, U):
        return pd.DataFrame(U[:self.grid_size].reshape(self.n_rows, self.n_columns))

    def show_action_value(self, Q):
        dictQ = [dict(zip(self.actions, np.round(values, 2)))
                 for values in Q[:self.grid_size, :]]

        return pd.DataFrame(np.array(dictQ).reshape(self.n_rows, self.n_columns))

    def show_policy(self, pi):
        policy = [self.actions[a] for a in pi[:self.grid_size]]
        return pd.DataFrame(np.array(policy).reshape(self.n_rows, self.n_columns))


class GridWorld5x5(GridWorld):
    def __init__(self, actions):
        super().__init__((5, 5), actions)

        for x in range(self.n_rows):
            for y in range(self.n_columns):
                s = self.xy2s(x, y)

                if x == 0 and y == 1:
                    self.R[s, :] = 10
                    self.T[s, :, self.xy2s(4, 1)] = 1
                elif x == 0 and y == 3:
                    self.R[s, :] = 5
                    self.T[s, :, self.xy2s(2, 3)] = 1
                else:
                    if x == 0 or x == self.n_rows-1:
                        if y == 0:
                            self.R[s, 0] = -1
                        elif y == self.n_columns - 1:
                            self.R[s, 2] = -1

                        if x == 0:
                            self.R[s, 1] = -1
                        elif x == self.n_rows - 1:
                            self.R[s, 3] = -1
                    elif y == 0:
                        self.R[s, 0] = -1
                    elif y == self.n_columns - 1:
                        self.R[s, 2] = -1

                    for a in range(self.n_actions):
                        if a == 0:
                            self.T[s, a, self.xy2s(x, y - 1)] = 1
                        elif a == 1:
                            self.T[s, a, self.xy2s(x - 1, y)] = 1
                        elif a == 2:
                            self.T[s, a, self.xy2s(x, y + 1)] = 1
                        elif a == 3:
                            self.T[s, a, self.xy2s(x + 1, y)] = 1

        self.check_model()


class GridWorld10x10(GridWorld):
    def __init__(self, actions):
        super().__init__((10, 10), actions, terminal=[(10, 0)])

        for x in range(self.n_rows):
            for y in range(self.n_columns):
                s = self.xy2s(x, y)

                if x == 2 and y == 7:
                    self.R[s, :] = 3
                    self.T[s, :, 100] = 1
                elif x == 7 and y == 8:
                    self.R[s, :] = 10
                    self.T[s, :, 100] = 1
                else:
                    if x == 7 and y == 3:
                        self.R[s, :] = -10
                    elif x == 4 and y == 3:
                        self.R[s, :] = -5
                    elif x == 0:
                        if y == 0 or y == 9:
                            self.R[s, :] = -0.2
                        else:
                            self.R[s, :] = -0.1
                        self.R[s, 1] = -0.7
                    elif x == 9:
                        if y == 0 or y == 9:
                            self.R[s, :] = -0.2
                        else:
                            self.R[s, :] = -0.1
                        self.R[s, 3] = -0.7
                    elif y == 0:
                        if x == 0 or x == 9:
                            self.R[s, :] = -0.2
                        else:
                            self.R[s, :] = -0.1
                        self.R[s, 0] = -0.7
                    elif y == 9:
                        if x == 0 or x == 9:
                            self.R[s, :] = -0.2
                        else:
                            self.R[s, :] = -0.1
                        self.R[s, 2] = -0.7

                    for a in range(self.n_actions):
                        if a == 0:
                            self.T[s, a, self.xy2s(x, y - 1)] += 0.7
                            self.T[s, a, self.xy2s(x, y + 1)] += 0.1
                            self.T[s, a, self.xy2s(x - 1, y)] += 0.1
                            self.T[s, a, self.xy2s(x + 1, y)] += 0.1
                        elif a == 1:
                            self.T[s, a, self.xy2s(x - 1, y)] += 0.7
                            self.T[s, a, self.xy2s(x + 1, y)] += 0.1
                            self.T[s, a, self.xy2s(x, y - 1)] += 0.1
                            self.T[s, a, self.xy2s(x, y + 1)] += 0.1
                        elif a == 2:
                            self.T[s, a, self.xy2s(x, y + 1)] += 0.7
                            self.T[s, a, self.xy2s(x, y - 1)] += 0.1
                            self.T[s, a, self.xy2s(x - 1, y)] += 0.1
                            self.T[s, a, self.xy2s(x + 1, y)] += 0.1
                        elif a == 3:
                            self.T[s, a, self.xy2s(x + 1, y)] += 0.7
                            self.T[s, a, self.xy2s(x - 1, y)] += 0.1
                            self.T[s, a, self.xy2s(x, y - 1)] += 0.1
                            self.T[s, a, self.xy2s(x, y + 1)] += 0.1

        self.R[0, 0] = -0.8
        self.R[90, 0] = -0.8
        self.R[0, 1] = -0.8
        self.R[9, 1] = -0.8
        self.R[9, 2] = -0.8
        self.R[99, 2] = -0.8
        self.R[90, 3] = -0.8
        self.R[99, 3] = -0.8

        self.check_model()


class GridWorld3x4(GridWorld):
    def __init__(self, actions):
        super().__init__((3, 4), actions, starts=[
            (2, 0)], terminal=[(3, 0)], walls=[(1, 1)])

        for x in range(self.n_rows):
            for y in range(self.n_columns):
                s = self.xy2s(x, y)

                if x == 0 and y == 3:
                    self.R[s, :] = 1
                    self.T[s, :, 12] = 1
                elif x == 1 and y == 3:
                    self.R[s, :] = -1
                    self.T[s, :, 12] = 1
                elif x == 1 and y == 1:
                    self.R[s, :] = 0
                else:
                    for a in range(self.n_actions):
                        if a == 0:
                            self.T[s, a, self.xy2s_wall(x, y - 1, s)] += 0.8
                            self.T[s, a, self.xy2s_wall(x - 1, y, s)] += 0.1
                            self.T[s, a, self.xy2s_wall(x + 1, y, s)] += 0.1
                        elif a == 1:
                            self.T[s, a, self.xy2s_wall(x - 1, y, s)] += 0.8
                            self.T[s, a, self.xy2s_wall(x, y - 1, s)] += 0.1
                            self.T[s, a, self.xy2s_wall(x, y + 1, s)] += 0.1
                        elif a == 2:
                            self.T[s, a, self.xy2s_wall(x, y + 1, s)] += 0.8
                            self.T[s, a, self.xy2s_wall(x - 1, y, s)] += 0.1
                            self.T[s, a, self.xy2s_wall(x + 1, y, s)] += 0.1
                        elif a == 3:
                            self.T[s, a, self.xy2s_wall(x + 1, y, s)] += 0.8
                            self.T[s, a, self.xy2s_wall(x, y - 1, s)] += 0.1
                            self.T[s, a, self.xy2s_wall(x, y + 1, s)] += 0.1

        self.check_model()


class RandomWalk(GridWorld):
    def __init__(self, actions):
        super().__init__((1, 5), actions, starts=[(0, 2)], terminal=[(0, 5)])

        self.R[4, 1] = 1

        for y in range(self.n_columns):
            if y == 0:
                self.T[y, 0, 5] = 1
                self.T[y, 1, y + 1] = 1
            elif y == 4:
                self.T[y, 0, y - 1] = 1
                self.T[y, 1, 5] = 1
            else:
                self.T[y, 0, y - 1] = 1
                self.T[y, 1, y + 1] = 1

        self.check_model()


class RandomWalk1000(GridWorld):
    def __init__(self, actions):
        super().__init__((1, 1000), actions, starts=[
            (0, 499)], terminal=[(0, 1000)])

        self.n_steps = 100

        rT = np.zeros((self.n_states, self.n_actions, self.n_states, 3))
        p = 1.0 / self.n_steps

        for y in range(self.n_columns):
            for a in range(self.n_actions):
                for step in range(1, self.n_steps + 1):
                    direction = 1
                    if a == 0:
                        direction = -1
                    newy = y + direction * step
                    if newy < 0:
                        rT[y, a, 1000, 0] += p
                    elif newy >= self.n_columns:
                        rT[y, a, 1000, 2] += p
                    else:
                        rT[y, a, newy, 1] += p

        self.T = np.sum(rT, axis=3)
        self.R = np.dot(np.sum(rT, axis=2), np.array([-1, 0, 1]))

        self.check_model()

    def step(self, s, a):
        n_step = np.random.randint(1, self.n_steps + 1)
        direction = 1
        if a == 0:
            direction = -1
        next_state = s + n_step * direction

        if next_state < 0:
            reward = -1
            next_state = 1000
        elif next_state >= 1000:
            reward = 1
            next_state = 1000
        else:
            reward = 0

        return next_state, reward, self.is_terminal(next_state)


class CliffWalking(GridWorld):
    def __init__(self, actions):
        super().__init__((4, 12), actions, starts=[(3, 0)], terminal=[
            (3, 11)], walls=[(3, i) for i in range(1, 11)])

        self.R = -1 * np.ones((self.n_states, self.n_actions))

        for x in range(self.n_rows):
            for y in range(self.n_columns):
                s = self.xy2s(x, y)

                if s in self.terminal or s in self.walls:
                    self.R[s, :] = 0
                else:
                    if x == 3 and y == 0:
                        self.R[s, 2] = -100
                        self.T[s, 0, self.starts[0]] = 1
                        self.T[s, 1, self.xy2s(x - 1, y)] = 1
                        self.T[s, 2, self.starts[0]] = 1
                        self.T[s, 3, self.starts[0]] = 1
                    elif x == 2 and y <= 10 and y >= 1:
                        self.R[s, 3] = -100
                        self.T[s, 0, self.xy2s(x, y - 1)] = 1
                        self.T[s, 1, self.xy2s(x - 1, y)] = 1
                        self.T[s, 2, self.xy2s(x, y + 1)] = 1
                        self.T[s, 3, self.starts[0]] = 1
                    else:
                        for a in range(self.n_actions):
                            if a == 0:
                                self.T[s, a, self.xy2s(x, y - 1)] = 1
                            elif a == 1:
                                self.T[s, a, self.xy2s(x - 1, y)] = 1
                            elif a == 2:
                                self.T[s, a, self.xy2s(x, y + 1)] = 1
                            elif a == 3:
                                self.T[s, a, self.xy2s(x + 1, y)] = 1

        self.check_model()


class AccessControl(GridWorld):
    def __init__(self, actions, n_servers, priorities):
        n_priorities = len(priorities)
        starts = [(p, n_servers) for p in range(n_priorities)]
        inf_actions = [[(p, 0), ['accept']] for p in range(n_priorities)]

        super().__init__((n_priorities, n_servers + 1), actions,
                         starts=starts, inf_actions=inf_actions)

        self.free_probability = 0.06
        self.priorities = priorities
        self.n_servers = n_servers

        for priority in range(self.n_rows):
            for n_free_servers in range(self.n_columns):
                s = self.xy2s(priority, n_free_servers)

                if n_free_servers != 0:
                    self.R[s, 1] = priorities[priority]

                    for a in range(self.n_actions):
                        busy_servers = n_servers - n_free_servers + a

                        self.T[s, a, :] = [binom.pmf(next_free_servers - n_free_servers + a, busy_servers,
                                                     self.free_probability) / n_priorities if next_free_servers >= n_free_servers - a else 0
                                           for next_priority in range(self.n_rows) for next_free_servers in range(self.n_columns)]
                else:
                    for a in range(self.n_actions):
                        if a == 0:
                            busy_servers = n_servers
                            self.T[s, a, :] = [binom.pmf(next_free_servers - n_free_servers + a, busy_servers,
                                                         self.free_probability) / n_priorities if next_free_servers >= n_free_servers - a else 0
                                               for next_priority in range(self.n_rows) for next_free_servers in range(self.n_columns)]

        self.check_model()

    def step(self, s, a):
        (_, n_free_servers) = self.s2xy(s)

        if n_free_servers > 0 and a == 1:
            n_free_servers -= 1
        busy_servers = self.n_servers - n_free_servers
        n_free_servers += np.random.binomial(busy_servers,
                                             self.free_probability)

        priority = random.choices(range(self.n_rows), k=1)[0]

        next_state = self.xy2s(priority, n_free_servers)

        return next_state, self.R[s, a], self.is_terminal(next_state)


class ShortCorridor(GridWorld):
    def __init__(self, actions):
        super().__init__((1, 3), actions, starts=[(0, 0)], terminal=[(0, 3)])

        for y in range(self.n_columns):
            self.R[y, :] = -1

        self.T[0, 0, 0] = 1
        self.T[0, 1, 1] = 1
        self.T[1, 0, 2] = 1
        self.T[1, 1, 0] = 1
        self.T[2, 0, 1] = 1
        self.T[2, 1, 3] = 1

        self.check_model()


class GridWorld1x3(GridWorld):
    def __init__(self, actions):
        super().__init__((1, 3), actions, terminal=[(0, 2)])

        self.R[0, :] = -1
        self.R[1, :] = -2

        self.T[0, 0, 0] = 0.2
        self.T[0, 0, 1] = 0.8
        self.T[0, 1, 0] = 0.9
        self.T[0, 1, 2] = 0.1
        self.T[1, 0, 1] = 0.2
        self.T[1, 0, 0] = 0.8
        self.T[1, 1, 1] = 0.9
        self.T[1, 1, 2] = 0.1

        self.check_model()
