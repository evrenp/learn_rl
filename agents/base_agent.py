import numpy as np


class BaseAgent(object):
    """
    Base Agent with memory per episode.
    """

    parameters = ['epsilon', 'gamma']

    def __init__(self,
                 env,
                 epsilon=0.5,
                 gamma=0.8,
                 max_n_steps=10000,
                 ):
        """Init

        Args:
            env ():
            epsilon (float): epsilon in epsilon-greedy action selection
            gamma (float): discount factor for future rewards
            max_n_steps (int): maximum number of steps per episode
        """
        self.epsilon = epsilon
        self.gamma = gamma
        self.max_n_steps = max_n_steps

        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self.observation_n_dim, self.observation_type, _ = self._get_index_utils(env.observation_space)
        self.action_n_dim, self.action_type, self.action_slice = self._get_index_utils(env.action_space)

        self.observation_path = None
        self.action_path = None
        self.reward_path = None
        self.t = None

        if self.observation_space.__class__.__name__ == 'Discrete':
            self.n_states = self.observation_space.n
        else:
            self.n_states = None

        if self.action_space.__class__.__name__ == 'Discrete':
            self.n_actions = self.action_space.n
        else:
            self.n_actions = None

        if (self.n_states is not None) and (self.n_actions is not None):
            self.q = np.zeros((self.n_states, self.n_actions))
        else:
            self.q = None



    @staticmethod
    def _get_index_utils(space):
        sample = space.sample()
        if type(sample) == np.ndarray:
            n_dim = sample.shape[0]
            assert sample.shape == (n_dim,)
            sample_type = type(sample[0])
            sample_slice = slice(None)
        elif type(sample) == int:
            n_dim = 1
            sample_type = int
            sample_slice = 0
        elif type(sample) == float:
            n_dim = 1
            sample_type = float
            sample_slice = 0
        else:
            raise ValueError()
        return n_dim, sample_type, sample_slice

    def get_parameters(self):
        return {key: getattr(self, key) for key in self.parameters}

    def get_parameters_as_str(self):
        return ', '.join(['{}={}'.format(k, v) for k, v in self.get_parameters().items()])

    def get_id(self):
        return '{}({})'.format(self.__class__.__name__, self.get_parameters_as_str())

    def set_time(self, t):
        self.t = t

    def reset_before_new_episode(self):
        """Resets agent before new episode

        Returns:
            None
        """
        self.observation_path = np.nan * np.zeros((self.max_n_steps, self.observation_n_dim))
        self.action_path = np.nan * np.zeros((self.max_n_steps, self.action_n_dim))
        self.reward_path = np.nan * np.zeros(self.max_n_steps)

    def act(self, observation, reward, done):
        """Selects action at t for moving into observation t+1.

        Args:
            observation (object): the observation at t
            reward (float): the reward at t for having taken the action at t-1
            done (bool): if the episode is done at t

        Returns:
            action (int, float, or np.array): the action selected at t, which brings the agent to the observation t+1
        """
        # update path variables
        self.observation_path[self.t, :] = observation
        self.reward_path[self.t] = reward

        # optional
        self.learn_at_t_before_action_selection()

        # choose action t and update path variable
        self.action_path[self.t, :] = self.select_action_at_t()

        # optional
        self.learn_at_t_after_action_selection()

        # at the end of the episode
        if done:
            # cut path variables
            self.observation_path = self.observation_path[:self.t + 1, :]
            self.action_path = self.action_path[:self.t + 1, :]
            self.reward_path = self.reward_path[:self.t + 1]

            # optional
            self.learn_at_last_t_of_episode()

        return self.action_path[self.t, self.action_slice].astype(self.action_type)

    def learn_at_t_before_action_selection(self):
        pass

    def learn_at_t_after_action_selection(self):
        pass

    def learn_at_last_t_of_episode(self):
        pass

    def select_action_at_t(self):
        """Selects action at t for moving into observation at t+1."""
        raise NotImplementedError()

    def select_epsilon_greedy_action_at_t(self):
        """Epsilon-greedy action selection of action at t

        Notes:
            - path variables for observation and reward at t are not nan, but action is nan

            - Definition of epsilon-greedy action selection:
                prob^pi(a|s) =  epsilon/m + 1 - epsiolon     if a = argmax_a'_over_A{ Q(s, a') }
                                epsilon/m                    otherwise
                with the number of actions m.

        Returns:
            action (int): action at time t
        """
        state_t = int(self.observation_path[self.t, 0])
        q_values = self.q[state_t, :]
        best_action = np.random.choice(
            np.arange(self.n_actions)[q_values == np.max(q_values)])
        pvals = self.epsilon / self.n_actions * np.ones(self.n_actions)
        pvals[best_action] += 1 - self.epsilon
        action = np.argwhere(np.random.multinomial(n=1, pvals=pvals))[0][0]
        return action