import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from gym.envs import make
from agents.base_agent import BaseAgent
from simulation.simulation import Simulation

ACTION_2_NAME = {0: 'noop', 1: 'fire', 2: 'right', 3: 'left', 4: 'fire_and_right', 5: 'fire_and_left'}
VALID_ACTIONS = [0, 1, 2, 3]
N_ACTIONS = 4
OBSERVATION_SHAPE = (210, 160, 3)
GRAY_IMAGE_SHAPE = (84, 84)
NETWORK_INPUT_SHAPE = (None, 84, 84, N_ACTIONS)  # where None represents n_samples_per_batch


class Estimator(object):
    """Neuronal network estimator

    This network is used for both the Q-Network and the Target Network.
    """

    def __init__(self, session, scope):
        assert scope in ['q', 'td_target']
        self.session = session
        self.scope = scope
        with tf.variable_scope(scope):
            self._build_graph()

    def _build_graph(self):
        # placeholder variables
        self.network_input = tf.placeholder(shape=NETWORK_INPUT_SHAPE, dtype=tf.uint8,
                                            name='network_input')  # the network_input at t-1
        self.target = tf.placeholder(shape=[None], dtype=tf.float32, name='target')  # the td_target, based on t
        self.action = tf.placeholder(shape=[None], dtype=tf.int32, name="action")  # the action selected at t-1

        # get n_samples_per_batch
        n_samples_per_batch = tf.shape(self.network_input)[0]

        # forward propagation in three-level network
        network_input_float = tf.to_float(self.network_input) / 255.0
        conv1 = tf.contrib.layers.conv2d(
            network_input_float, 32, 8, 4, activation_fn=tf.nn.relu)
        conv2 = tf.contrib.layers.conv2d(
            conv1, 64, 4, 2, activation_fn=tf.nn.relu)
        conv3 = tf.contrib.layers.conv2d(
            conv2, 64, 3, 1, activation_fn=tf.nn.relu)

        # fully connected output layers
        flattened = tf.contrib.layers.flatten(conv3)
        fc1 = tf.contrib.layers.fully_connected(flattened, 512)
        self.predictions_for_all_actions = tf.contrib.layers.fully_connected(fc1, len(VALID_ACTIONS))

        # get the predictions for the chosen actions only
        gather_indices = tf.range(n_samples_per_batch) * tf.shape(self.predictions_for_all_actions)[1] + self.action
        self.prediction_for_selected_action = tf.gather(tf.reshape(self.predictions_for_all_actions, [-1]),
                                                        gather_indices)

        # calculate the loss
        self.losses = tf.squared_difference(self.target, self.prediction_for_selected_action)
        self.loss = tf.reduce_mean(self.losses)

        # Optimizer Parameters from original paper
        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        self.train_op = self.optimizer.minimize(self.loss)  # maybe add global step

        self.init_op = tf.global_variables_initializer()


class PreProcessor(object):
    def __init__(self, session):
        self.session = session

        with tf.variable_scope('pre_processor'):
            self.observation_placeholder = tf.placeholder(shape=OBSERVATION_SHAPE, dtype=tf.uint8)
            self.gray_image = tf.image.rgb_to_grayscale(self.observation_placeholder)
            self.gray_image = tf.image.crop_to_bounding_box(self.gray_image, 34, 0, 160, 160)
            self.gray_image = tf.image.resize_images(
                self.gray_image, GRAY_IMAGE_SHAPE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.gray_image = tf.squeeze(self.gray_image)

            self.init_op = tf.global_variables_initializer()

    def _single_observation_2_network_input(self, observation):
        """Process a single observation_placeholder from the environment

        Args:
            observation (np.array, shape=OBSERVATION_SHAPE): Atari RGB observation_placeholder

        Returns:
            network_input (np.array, shape=NETWORK_INPUT_SHAPE): grayscale gray_image duplicated for each action
        """
        gray_image_values = self.session.run(self.gray_image, feed_dict={self.observation_placeholder: observation})
        network_input = np.stack([gray_image_values] * N_ACTIONS, axis=2)
        network_input = np.array([network_input])  # format of batch with single sample
        return network_input

    def observation_list_2_network_input(self, observation_list):
        network_input = np.vstack([self._single_observation_2_network_input(observation=observation) for observation in
                                   observation_list])
        return network_input


class DeepQLearningAgent(BaseAgent):
    def __init__(self, session, **kwargs):

        self.session = session
        self.pre_processor = PreProcessor(session=self.session)
        self.q_estimator = Estimator(session=self.session, scope='q')
        self.td_target_estimator = Estimator(session=self.session, scope='td_target')

        super(DeepQLearningAgent, self).__init__(**kwargs)

        # overwrite n_actions
        self.n_actions = N_ACTIONS

        # caching variable
        self.q_values_of_possible_actions_at_t = np.zeros(self.n_actions)

    def tf_init_at_session_start(self):
        for init_op in [self.pre_processor.init_op, self.q_estimator.init_op, self.td_target_estimator.init_op]:
            self.session.run(init_op)

    def predict_q_values_of_possible_actions_at_t(self, observation_list):
        """Predicts with td_target_estimator

        Args:
            observation_list (list): list of single-sample observations at t-1

        Returns:
            predictions_for_all_actions (np.array of shape (n_samples_per_patch, n_actions))
        """

        network_input = self.pre_processor.observation_list_2_network_input(observation_list=observation_list)

        predictions_for_all_actions = self.session.run(self.td_target_estimator.predictions_for_all_actions,
                                                       {self.td_target_estimator.network_input: network_input})

        return predictions_for_all_actions

    def fit(self, observation_list, action_list, td_target_list):
        """Fits q_estimator"""

        network_input = self.pre_processor.observation_list_2_network_input(observation_list=observation_list)

        feed_dict = {self.q_estimator.network_input: network_input, self.q_estimator.target: td_target_list,
                     self.q_estimator.action: action_list}
        _, loss = self.session.run([self.q_estimator.train_op, self.q_estimator.loss], feed_dict)
        return loss

    def learn_at_t_before_action_selection(self):
        if self.t > 0:

            # TODO: add actual batch-learning schedule

            # just a test
            self.q_values_of_possible_actions_at_t = self.predict_q_values_of_possible_actions_at_t(
                observation_list=[self.observations[self.t - 1]])[0, :]

            # td_target
            td_target = self.rewards[self.t] + self.gamma * np.max(self.q_values_of_possible_actions_at_t)

            # test fit for single-sample batch
            loss = self.fit(observation_list=[self.observations[self.t - 1]], action_list=[self.actions[self.t - 1]],
                            td_target_list=[td_target])

    def select_action_at_t(self):
        return self.select_epsilon_greedy_action_at_t(
            q_values_of_possible_actions_at_t=self.q_values_of_possible_actions_at_t)


def run_agent():
    session = tf.Session(graph=tf.get_default_graph())
    env = make("Breakout-v0")
    env._max_episode_steps = 40
    agent = DeepQLearningAgent(session=session, env=env, epsilon=0.5, gamma=1.0)
    with agent.session:
        agent.tf_init_at_session_start()
        simulation = Simulation(env=env, agent=agent, is_render=False)
        simulation.simulate_episodes(n_episodes=2)
        simulation.terminate()


if __name__ == '__main__':
    run_agent()
