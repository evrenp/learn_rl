import numpy as np

import tensorflow as tf
from gym import make
from simulation.simulation import Simulation
from agents.sarsa_max_function_approximation_sklearn import SarsaMaxFunctionApproximationAgentSklearn


class SarsaMaxFunctionApproximationAgentTF(SarsaMaxFunctionApproximationAgentSklearn):
    def __init__(self,
                 **kwargs
                 ):
        """Init

        Args:
            kwargs (dict): kwargs of SarsaMaxFunctionApproximationAgentSklearn
        """
        self.session = tf.Session()
        super(SarsaMaxFunctionApproximationAgentTF, self).__init__(**kwargs)

    def _get_estimators(self):
        """One estimator per action"""
        estimators = []
        for action in range(self.n_actions):
            estimator = SingleActionGraph(session=self.session, action=action, n_features=self.n_features)
            estimators.append(estimator)
        return estimators

    def tf_init_at_session_start(self):
        for action in range(self.n_actions):
            self.session.run(self.estimators[action].init_op)


class SingleActionGraph(object):
    """Mimics the sklearn estimator for SGD regression used in sklearn version."""

    def __init__(self, session, action, n_features):
        self.session = session

        with tf.variable_scope('action_{}'.format(action)):
            self.features = tf.placeholder(tf.float32, shape=[None, n_features], name='features')
            self.td_target = tf.placeholder(tf.float32, shape=[None, 1], name='td_target')

            weights = tf.Variable(tf.random_uniform([n_features, 1], -1.0, 1.0), name='weights')
            self.y_hat = tf.matmul(self.features, weights, name='y_hat')

            self.loss = tf.reduce_mean(tf.square(self.td_target - self.y_hat), name='loss')
            optimizer = tf.train.GradientDescentOptimizer(0.01)
            self.train = optimizer.minimize(self.loss, name='train')

            # init_op per scope
            self.init_op = tf.global_variables_initializer()

    def predict(self, X):
        y_hat = np.array([self.session.run(self.y_hat, {self.features: X})[0, 0]])  # np.array because of sklearn format
        return y_hat

    def partial_fit(self, X, y):
        self.session.run(self.train, {self.features: X, self.td_target: np.array([y]).reshape((1, 1))})


if __name__ == '__main__':
    env = make('MountainCar-v0')
    env._max_episode_steps = 400
    agent = SarsaMaxFunctionApproximationAgentTF(env=env, epsilon=0.5, gamma=1.0)
    with agent.session:
        agent.tf_init_at_session_start()
        agent = SarsaMaxFunctionApproximationAgentSklearn(env=env, epsilon=0.5, gamma=1.0)
        simulation = Simulation(env=env, agent=agent, is_render=False)
        simulation.simulate_episodes(n_episodes=20)
        simulation.is_render = True
        simulation.simulate_episodes(n_episodes=5)
        simulation.terminate()