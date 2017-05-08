import numpy as np
from agents.base_agent import BaseAgent


class SarsaMaxAgent(BaseAgent):
    """Q-learning algorithm, also known as SARSA-MAX off-policy learning algorithm.

    Algorithm:
        - The agent follows an epsilon-greedy behavior policy.

        - The Q-values are updated according to a fully greedy alternative policy (max over possible actions a')

            Q(S_t-1, A_t-1) <- Q(S_t-1, A_t-1) + alpha * delta

            with prediction_error delta = (R_t + gamma * max_over_a'{ Q(S_t, a')} - Q(S_t-1, A_t-1))
    """
    parameters = ['epsilon', 'gamma', 'alpha']

    def __init__(self,
                 alpha=0.1,
                 **kwargs
                 ):
        """Init

        Args:
            alpha (float): learning rate for updating q after each step
            kwargs (dict): kwargs of DiscreteActionsAgent
        """
        super(SarsaMaxAgent, self).__init__(**kwargs)

        assert self.action_space.__class__.__name__ == 'Discrete', 'Only works for discrete action space'
        assert self.observation_space.__class__.__name__ == 'Discrete', 'Only works for discrete observation space'

        self.alpha = alpha

        self.q_values_of_possible_actions_at_t = np.zeros(self.n_actions)

    def learn_at_t_before_action_selection(self):
        if self.t > 0:
            # get index
            state_t_minus_1 = self.observations[self.t - 1]
            action_t_minus_1 = self.actions[self.t - 1]
            state_t = self.observations[self.t]

            self.q_values_of_possible_actions_at_t = self.q[state_t, :]

            # td_target
            td_target = self.rewards[self.t] + self.gamma * np.max(self.q_values_of_possible_actions_at_t)

            # prediction_error
            prediction_error = td_target - self.q[state_t_minus_1, action_t_minus_1]

            # update q
            self.q[state_t_minus_1, action_t_minus_1] += self.alpha * prediction_error

    def select_action_at_t(self):
        """Select action at time t"""
        return self.select_epsilon_greedy_action_at_t(q_values_of_possible_actions_at_t=self.q_values_of_possible_actions_at_t)


class MonteCarloAgent(BaseAgent):
    """MonteCarlo Agent

    Algorithm:

        - Run each episode under epsilon-greedy policy (creates one Monte-Carlo-Sample).

        - At the end of each episode:

        - For each (t, S_t, A_t) in the episode:

            - Compute total discounted future_return G_t for each t.

            - Update state_action_counter of visited state action pairs: N(S_t, A_t) <- N(S_t, A_t) + 1

            - Update q-values: Q(S_t, A_t) <- Q(S_t, A_t) + alpha * (G_t - Q(S_t, A_t))

                with a experience-dependent decaying learning rate alpha = 1/N(S_t, A_t)
    """
    parameters = ['epsilon', 'gamma']

    def __init__(self,
                 **kwargs
                 ):
        """Init

        Args:
            kwargs (dict): kwargs of DiscreteQAgent
        """
        super(MonteCarloAgent, self).__init__(**kwargs)

        assert self.action_space.__class__.__name__ == 'Discrete', 'Only works for discrete action space'
        assert self.observation_space.__class__.__name__ == 'Discrete', 'Only works for discrete observation space'

        self.state_action_counter = np.zeros((self.n_states, self.n_actions))

    def learn_at_last_t_of_episode(self):

        for t, (state, action) in enumerate(zip(self.observations, self.actions)):

            # increment counter
            self.state_action_counter[state, action] += 1

            # future return target
            future_return_target = np.sum(self.gamma ** np.arange(len(self.rewards[t + 1:])) * self.rewards[t + 1:])

            # dynamic learning rate
            alpha = 1. / self.state_action_counter[state, action]

            # prediction_error
            prediction_error = future_return_target - self.q[state, action]

            # update q
            self.q[state, action] += alpha * prediction_error

    def select_action_at_t(self):
        """Select action at time t"""
        state_t = self.observations[self.t]
        return self.select_epsilon_greedy_action_at_t(q_values_of_possible_actions_at_t=self.q[state_t, :])


class SarsaLambdaAgent(BaseAgent):
    """Backward-view Sarsa(lambda) algorithm

    - Algorithm

        - Actions are selected under epsilon-greedy policy.

        - Eligibility traces allow information to flow from current reward backwards to eligible states.

        - For each time t in episode

            - Increment eligibility E(S_t, A_t) <- E(S_t, A_t) + 1

            - Prediction error delta_t = R_t+1 + gamma * Q(S_t+1, A_t+1) - Q(S_t, A_t)

            - For all states s and actions a update eligibility and q:

                Q(s, a) <- Q(s, a) + alpha * delta_t * E(s, a)

                E(s, a) <- gamma * lambda * E(s, a)   (lambda is a decay rate for eligibility of remote states)
    """
    params = ['epsilon', 'gamma', 'alpha', 'lam']

    def __init__(self,
                 alpha=0.05,
                 lam=0.9,
                 **kwargs):
        """Init

        Args:
            gamma (float): discount factor for future rewards
            alpha (float): learning rate for updating q after each step
            lam (float): memory factor of eligibility traces
            kwargs (dict): kwargs of DiscreteQAgent
        """
        super(SarsaLambdaAgent, self).__init__(**kwargs)

        assert self.action_space.__class__.__name__ == 'Discrete', 'Only works for discrete action space'
        assert self.observation_space.__class__.__name__ == 'Discrete', 'Only works for discrete observation space'

        self.lam = lam
        self.alpha = alpha

        # dynamic variables for whole simulation
        self.eligibility = np.zeros((self.n_states, self.n_actions))

    def learn_at_t_after_action_selection(self):
        """action at time t is already defined."""
        if self.t > 0:
            # get index
            state_t = self.observations[self.t]
            action_t = self.actions[self.t]
            state_t_minus_1 = self.observations[self.t - 1]
            action_t_minus_1 = self.actions[self.t - 1]

            # increment eligibility trace
            self.eligibility[state_t_minus_1, action_t_minus_1] += 1

            # td_target
            td_target = self.rewards[self.t] + self.gamma * self.q[state_t, action_t]

            # prediction_error
            prediction_error = td_target - self.q[state_t_minus_1, action_t_minus_1]

            # update q
            self.q += self.alpha * prediction_error * self.eligibility

            # update eligibility
            self.eligibility *= self.gamma * self.lam

    def select_action_at_t(self):
        """Select action at time t"""
        state_t = self.observations[self.t]
        return self.select_epsilon_greedy_action_at_t(q_values_of_possible_actions_at_t=self.q[state_t, :])
