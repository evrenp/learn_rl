from gym import Env, spaces
from gym.utils import seeding
import matplotlib.pyplot as plt
import numpy as np


class GridEnv(Env):
    metadata = {}

    def __init__(self, width=6, height=3):
        self.action_space = spaces.Discrete(4)
        self.width = width
        self.height = height
        self.n_states = self.width * self.height
        self.observation_space = spaces.Discrete(self.n_states)
        self.state = None
        self.action_2_name = {0: 'left', 1: 'right', 2: 'up', 3: 'down'}
        self.action_2_move = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

        # immediate reward for entering a state
        self.terminal_state = self.n_states - 1
        self.r = -np.ones(self.n_states)
        self.r[self.terminal_state] = 0

    def _state_2_x_y(self, state):
        x = int(state % self.width)
        y = int((state - x) / self.width)
        return x, y

    def _x_y_2_state(self, x, y):
        return int(y * self.width + x)

    def _is_inside_maze(self, x, y):
        return (x >= 0) and (x < self.width) and (y >= 0) and (y < self.height)

    def reward_for_entering_state(self, state):
        return self.r[state]

    def state_action_2_new_state(self, state, action):
        x, y = self._state_2_x_y(state)
        dx, dy = self.action_2_move[action]
        if self._is_inside_maze(x + dx, y + dy):
            x_new, y_new = x + dx, y + dy
        else:
            x_new, y_new = x, y
        new_state = self._x_y_2_state(x_new, y_new)
        return new_state

    def _step(self, action):
        """Makes a step.

        Args:
            action (int): the action at t by the agent

        Returns:
            observation (int): the observation at t+1
            reward (int): the reward collected at t+1 for having taken action at t
            done (bool): whether the agent is in a terminal state at t+1
        :param action:
        :return:
        """

        new_state = self.state_action_2_new_state(state=self.state, action=action)
        reward = self.reward_for_entering_state(state=new_state)

        done = False
        if new_state == self.terminal_state:
            done = True

        self.state = new_state
        observation = self.state

        info = None

        return observation, reward, done, info

    def _reset(self):
        # reset_before_new_episode state
        self.state = 0

        # return observation of self.state
        return self.state

    # def _render(self, mode='rgb_array', close=False):  #rgb_array: Return an numpy.ndarray with shape (x, y, 3)
    #     pass

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def print_greedy_policy(self, greedy_policy):
        print(np.array([self.action_2_name[action].rjust(5) for action in greedy_policy]).reshape(
            (self.height, self.width)))


def plot_trajectory(env, agent):
    import matplotlib.pyplot as plt
    n_steps = agent.observation_path.shape[0]
    track = np.zeros((n_steps, 2))
    for t in (range(n_steps)):
        track[t, :] = env._state_2_x_y(state=agent.observation_path[t, 0])

    plt.figure(figsize=(env.width, env.height))
    plt.plot(track[:, 0], track[:, 1], 'r-')
    plt.plot(track[0, 0], track[0, 1], 'ro')
    plt.plot(track[-1, 0], track[-1, 1], 'rs')
    plt.show()
