import numpy as np

from agents.base_agent import BaseAgent

from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler


class FunctionApproximationAgent(BaseAgent):
    hyper_parameters = ['epsilon', 'gamma']

    def __init__(self,
                 **kwargs
                 ):
        """Init

        Args:
            kwargs (dict): kwargs of BaseAgent
        """
        super(FunctionApproximationAgent, self).__init__(**kwargs)

        assert self.action_space.__class__.__name__ == 'Discrete', 'Only works for discrete action space'
        assert self.observation_space.__class__.__name__ == 'Box', 'Only works for Box observation space'

        self.scaler, self.featurizer = self._get_scaler_and_featurizer()
        self.estimators = self._get_estimators()
        self.q_values_of_possible_actions_at_t = np.zeros(self.n_actions)

    def _get_scaler_and_featurizer(self):
        scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
        observations = np.array([self.observation_space.sample() for _ in range(10000)])
        scaler.fit(X=observations)
        # Used to converte a state to a featurizes represenation.
        # We use RBF kernels with different variances to cover different parts of the space
        featurizer = FeatureUnion([
            ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
            ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
            ("rbf4", RBFSampler(gamma=0.5, n_components=100))
        ])
        featurizer.fit(scaler.transform(observations))
        return scaler, featurizer

    def _get_estimators(self):
        """One estimator per action"""
        estimators = []
        for _ in range(self.n_actions):
            estimator = SGDRegressor(learning_rate="constant")
            fake_observation = np.zeros(self.observation_n_dim)
            fake_target = np.zeros(1)
            features = self.observation_2_features(fake_observation)
            estimator.partial_fit(features, fake_target)
            estimators.append(estimator)
        return estimators

    def observation_2_features(self, observation):
        scaled_observation = self.scaler.transform(X=observation.reshape((1, self.observation_n_dim)))
        features = self.featurizer.transform(scaled_observation)
        return features

    def predict_q_value(self, observation, action):
        features = self.observation_2_features(observation)
        q_hat = self.estimators[action].predict(features)[0]
        return q_hat

    def learn_at_t_before_action_selection(self):
        if self.t > 0:
            # features for prediction of q at t-1
            features_at_t_minus_1 = self.observation_2_features(observation=self.observation_path[self.t - 1, :])

            self.q_values_of_possible_actions_at_t = np.array(
                [self.predict_q_value(observation=self.observation_path[self.t, :], action=action) for action
                 in range(self.n_actions)])

            # td_target
            td_target = self.reward_path[self.t] + self.gamma * np.max(self.q_values_of_possible_actions_at_t)

            # partial fit
            action_t_minus_1 = int(self.action_path[self.t - 1, 0])
            self.estimators[action_t_minus_1].partial_fit(X=features_at_t_minus_1, y=np.array([td_target]))

    def select_action_at_t(self):
        return self.select_epsilon_greedy_action_at_t(
            q_values_of_possible_actions_at_t=self.q_values_of_possible_actions_at_t)