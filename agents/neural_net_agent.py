import numpy as np

import tensorflow as tf
from gym import make
from simulation.simulation import Simulation
from agents.function_approximation_agents import SarsaMaxFunctionApproximationAgent

from agents.base_agent import BaseAgent


class SarsaMaxFunctionApproximationAgentTF(BaseAgent):
    parameters = ['epsilon', 'gamma']

    def __init__(self,
                 save_monitor=False,
                 **kwargs
                 ):
        """Init

        Args:
            kwargs (dict): kwargs of BaseAgent
        """
        super(SarsaMaxFunctionApproximationAgentTF, self).__init__(**kwargs)

        assert self.action_space.__class__.__name__ == 'Discrete', 'Only works for discrete action space'
        assert self.observation_space.__class__.__name__ == 'Box', 'Only works for Box observation space'
        assert len(self.observation_space.sample().shape) == 1

        self.scaler, self.featurizer = self._get_scaler_and_featurizer()
        self.n_features = self.observation_2_features(observation=self.observation_space.sample()).shape[1]

        self.session = tf.Session()

        self.estimators = self._get_estimators()
        self.q_values_of_possible_actions_at_t = np.zeros(self.n_actions)

        self.save_monitor = save_monitor
        if save_monitor:
            self.monitor = {'features': [], 'td_target': [], 'action': [], 'y_hat': []}

    _get_scaler_and_featurizer = SarsaMaxFunctionApproximationAgent._get_scaler_and_featurizer

    observation_2_features = SarsaMaxFunctionApproximationAgent.observation_2_features

    select_action_at_t = SarsaMaxFunctionApproximationAgent.select_action_at_t

    def _get_estimators(self):
        """One estimator per action"""
        estimators = []
        for action in range(self.n_actions):
            estimator = SingleActionGraph(session=self.session, action=action, n_features=self.n_features)
            estimators.append(estimator)
        return estimators

    def predict_q_value(self, observation, action):
        features = self.observation_2_features(observation)
        q_hat = self.estimators[action].predict(session=self.session, features=features)
        return q_hat

    def learn_at_t_before_action_selection(self):
        if self.t > 0:
            # features for prediction of q at t-1
            features_at_t_minus_1 = self.observation_2_features(observation=self.observations[self.t - 1])

            self.q_values_of_possible_actions_at_t = np.array(
                [self.predict_q_value(observation=self.observations[self.t], action=action) for action
                 in range(self.n_actions)])

            # td_target
            td_target = self.rewards[self.t] + self.gamma * np.max(self.q_values_of_possible_actions_at_t)

            # partial fit
            action_t_minus_1 = self.actions[self.t - 1]
            self.estimators[action_t_minus_1].fit_single_sample(session=self.session, features=features_at_t_minus_1,
                                                                td_target=td_target)

            if self.save_monitor:
                self.monitor['features'].append(features_at_t_minus_1)
                self.monitor['action'].append(action_t_minus_1)
                self.monitor['td_target'].append(td_target)
                self.monitor['y_hat'].append(self.estimators[action_t_minus_1].predict(X=features_at_t_minus_1))


class SingleActionGraph(object):
    def __init__(self, session, action, n_features):
        assert action in range(3)

        with tf.variable_scope('action_{}'.format(action)):
            self.features = tf.placeholder(tf.float32, shape=[None, n_features], name='features')
            self.td_target = tf.placeholder(tf.float32, shape=[None, 1], name='td_target')

            weights = tf.Variable(tf.random_uniform([n_features, 1], -1.0, 1.0), name='weights')
            self.y_hat = tf.matmul(self.features, weights, name='y_hat')

            self.loss = tf.reduce_mean(tf.square(self.td_target - self.y_hat), name='loss')
            optimizer = tf.train.GradientDescentOptimizer(0.01)
            self.train = optimizer.minimize(self.loss, name='train')

            self.init_op = tf.initialize_all_variables()

    def predict(self, session, features):
        """Predict q-value of single observation

        Args:
            session
            features

        Returns:
            y_hat (float)
        """
        y_hat = session.run(self.y_hat, {self.features: features})[0, 0]
        return y_hat

    def fit_single_sample(self, session, features, td_target):
        session.run(self.train, {self.features: features, self.td_target: np.array([td_target]).reshape((1, 1))})


if __name__ == '__main__':

    env = make('MountainCar-v0')
    env._max_episode_steps = 400
    agent = SarsaMaxFunctionApproximationAgentTF(env=env, epsilon=0.5, gamma=1.0)
    with agent.session:
        for action in range(3):
            agent.session.run(agent.estimators[action].init_op)
        simulation = Simulation(env=env, agent=agent, logger_level='INFO', is_render=False, max_n_steps=5000)
        simulation.simulate_episodes(n_episodes=40)
        simulation.is_render = True
        simulation.simulate_episodes(n_episodes=5)
        simulation.terminate()
