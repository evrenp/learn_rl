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
                 n_past_episodes_in_memory=0,
                 ):
        """Init

        Args:
            env (gym.Env): environment
            epsilon (float): epsilon in epsilon-greedy action selection
            gamma (float): discount factor for future rewards
            max_n_steps (int): maximum number of steps per episode
            n_past_episodes_in_memory (int): number of past episodes in addition to current one to be kept in memory
        """
        self.epsilon = epsilon
        self.gamma = gamma
        self.max_n_steps = max_n_steps
        self.n_past_episodes_in_memory = n_past_episodes_in_memory

        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.n_states, self.n_actions, self.q = self._get_utils_for_discrete_spaces()

        self.observations, self.actions, self.rewards = self._init_path_variables()
        self.t = None

        # memory of path_variables
        self.past_observations = []
        self.past_actions = []
        self.past_rewards = []

    def _init_path_variables(self):
        """Init path variables of current episode"""
        observations = self.max_n_steps * [None]
        actions = self.max_n_steps * [None]
        rewards = np.nan * np.zeros(self.max_n_steps)
        return observations, actions, rewards

    def _get_utils_for_discrete_spaces(self):
        if self.observation_space.__class__.__name__ == 'Discrete':
            n_states = self.observation_space.n
        else:
            n_states = None
        if self.action_space.__class__.__name__ == 'Discrete':
            n_actions = self.action_space.n
        else:
            n_actions = None
        if (n_states is not None) and (n_actions is not None):
            q = np.zeros((n_states, n_actions))
        else:
            q = None
        return n_states, n_actions, q

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
        self.observations, self.actions, self.rewards = self._init_path_variables()

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
        self.observations[self.t] = observation
        self.rewards[self.t] = reward

        # optional
        self.learn_at_t_before_action_selection()

        # choose action t and update path variable
        self.actions[self.t] = self.select_action_at_t()

        # optional
        self.learn_at_t_after_action_selection()

        # at the end of the episode
        if done:
            # cut path variables
            self.observations = self.observations[:self.t + 1]
            self.actions = self.actions[:self.t + 1]
            self.rewards = self.rewards[:self.t + 1]

            # optional
            self.learn_at_last_t_of_episode()

            # write path_variables to memory
            if self.n_past_episodes_in_memory > 0:
                self.past_observations.append(self.observations)
                self.past_actions.append(self.actions)
                self.past_rewards.append(self.rewards)

                # cut memory
                if len(self.past_observations) > self.n_past_episodes_in_memory:
                    del self.past_observations[0]
                    del self.past_actions[0]
                    del self.past_rewards[0]


        return self.actions[self.t]

    def learn_at_t_before_action_selection(self):
        pass

    def learn_at_t_after_action_selection(self):
        pass

    def learn_at_last_t_of_episode(self):
        pass

    def select_action_at_t(self):
        """Selects action at t for moving into observation at t+1."""
        raise NotImplementedError()

    def select_epsilon_greedy_action_at_t(self, q_values_of_possible_actions_at_t):
        """Epsilon-greedy action selection of action at t

        Args:
            q_values_of_possible_actions_at_t (np.array): q-values of possible actions at t before action selection

        Notes:
            - Definition of epsilon-greedy action selection:
                prob^pi(a|s) =  epsilon/m + 1 - epsiolon     if a = argmax_a'_over_A{ Q(s, a') }
                                epsilon/m                    otherwise
                with the number of actions m.

        Returns:
            action (int): action at time t
        """
        assert q_values_of_possible_actions_at_t.shape == (self.n_actions,)

        best_action = np.random.choice(
            np.arange(self.n_actions)[q_values_of_possible_actions_at_t == np.max(q_values_of_possible_actions_at_t)])
        pvals = self.epsilon / self.n_actions * np.ones(self.n_actions)
        pvals[best_action] += 1 - self.epsilon
        action = np.argwhere(np.random.multinomial(n=1, pvals=pvals))[0][0]
        return action
