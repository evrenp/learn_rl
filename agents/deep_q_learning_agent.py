import os
import numpy as np
import matplotlib.pyplot as plt
from random import sample
import tensorflow as tf
from gym.envs import make
from agents.base_agent import BaseAgent
from simulation.simulation import Simulation, get_logger

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
    def __init__(
            self,
            session,
            n_samples_replay_memory=300,  # 500000
            n_batch_samples=32,
            copy_interval=100,  # 10000
            logger_level='INFO',
            **kwargs
    ):

        self.logger = get_logger(level=logger_level, name='DeepQLearningAgent')
        self.session = session
        self.n_samples_replay_memory = n_samples_replay_memory
        self.n_batch_samples = n_batch_samples
        self.copy_interval = copy_interval
        self.pre_processor = PreProcessor(session=self.session)
        self.q_estimator = Estimator(session=self.session, scope='q')
        self.td_target_estimator = Estimator(session=self.session, scope='td_target')

        super(DeepQLearningAgent, self).__init__(**kwargs)

        # overwrite n_actions
        self.n_actions = N_ACTIONS

        # caching variable
        self.q_values_of_possible_actions_at_t = np.zeros(self.n_actions)

        self.replay_memory = []

        self.total_t = 0

    def tf_init_at_session_start(self):
        for init_op in [self.pre_processor.init_op, self.q_estimator.init_op, self.td_target_estimator.init_op]:
            self.session.run(init_op)

    def predict_q_values_of_possible_actions_at_t(self, observation_at_t_list):
        """Predicts with td_target_estimator

        Args:
            observation_at_t_list (list): list of single-sample observations at t

        Returns:
            predictions_for_all_actions (np.array of shape (n_samples_per_patch, n_actions))
        """

        network_input = self.pre_processor.observation_list_2_network_input(observation_list=observation_at_t_list)

        predictions_for_all_actions = self.session.run(self.td_target_estimator.predictions_for_all_actions,
                                                       {self.td_target_estimator.network_input: network_input})

        return predictions_for_all_actions

    def fit(self, observation_at_t_minus_1_list, action_at_t_minus_1_list, td_target_at_t_list):
        """Fits q_estimator

        Args:
            observation_at_t_minus_1_list (list): list of single sample observations at t-1  Rename to observation_at_t_minus_1_list
        """

        network_input = self.pre_processor.observation_list_2_network_input(
            observation_list=observation_at_t_minus_1_list)

        feed_dict = {self.q_estimator.network_input: network_input, self.q_estimator.target: td_target_at_t_list,
                     self.q_estimator.action: action_at_t_minus_1_list}
        _, loss = self.session.run([self.q_estimator.train_op, self.q_estimator.loss], feed_dict)
        return loss

    def learn_at_t_before_action_selection(self):

        if self.t > 0:

            # needed in select_action_at_t
            self.q_values_of_possible_actions_at_t = self.predict_q_values_of_possible_actions_at_t(
                observation_at_t_list=[self.observations[self.t]])[0, :]

            # always append replay memory
            self.replay_memory.append((
                self.observations[self.t - 1],
                self.observations[self.t],
                self.actions[self.t - 1],
                self.rewards[self.t]
            ))

            # learn only if replay memory has desired size
            if len(self.replay_memory) == self.n_samples_replay_memory + 1:
                self.replay_memory.pop(0)

                training_batch = sample(self.replay_memory, self.n_batch_samples)

                observation_at_t_minus_1_list, observation_at_t_list, action_at_t_minus_1_list, reward_at_t_list = map(
                    list, zip(*training_batch))

                q_values_of_possible_actions_at_t_array = self.predict_q_values_of_possible_actions_at_t(
                    observation_at_t_list=observation_at_t_list)

                td_target_at_t_list = list(np.array(reward_at_t_list) + self.gamma * np.max(
                    q_values_of_possible_actions_at_t_array, axis=1))

                loss = self.fit(
                    observation_at_t_minus_1_list=observation_at_t_minus_1_list,
                    action_at_t_minus_1_list=action_at_t_minus_1_list,
                    td_target_at_t_list=td_target_at_t_list
                )

                # update the target estimator
                if self.total_t % self.copy_interval == 0:
                    self.copy_model_parameters()
                    self.logger.debug('Copied model parameters from q_estimator to td_target_estimator network.')
                    self.logger.debug('Loss={:.4f} at total_t={}'.format(loss, self.total_t))

        self.total_t += 1

    def select_action_at_t(self):
        return self.select_epsilon_greedy_action_at_t(
            q_values_of_possible_actions_at_t=self.q_values_of_possible_actions_at_t)

    def copy_model_parameters(self):
        q_estimator_params = [t for t in tf.trainable_variables() if t.name.startswith(self.q_estimator.scope)]
        q_estimator_params = sorted(q_estimator_params, key=lambda v: v.name)
        td_target_estimator_params = [t for t in tf.trainable_variables() if
                                      t.name.startswith(self.td_target_estimator.scope)]
        td_target_estimator_params = sorted(td_target_estimator_params, key=lambda v: v.name)
        update_ops = []
        for q_estimator_param, td_estimator_param in zip(q_estimator_params, td_target_estimator_params):
            op = td_estimator_param.assign(q_estimator_param)
            update_ops.append(op)
        self.session.run(update_ops)


def run_agent():
    session = tf.Session(graph=tf.get_default_graph())
    env = make("Breakout-v0")
    # env._max_episode_steps = 500
    agent = DeepQLearningAgent(
        session=session,
        env=env,
        epsilon=0.5,
        gamma=0.99,
        n_samples_replay_memory=2000,
        n_batch_samples=32,
        copy_interval=1000,
        logger_level='DEBUG'
    )

    with agent.session:
        agent.tf_init_at_session_start()
        simulation = Simulation(env=env, agent=agent, is_render=False, logger_level='DEBUG')

        for _ in range(100):
            simulation.simulate_episodes(n_episodes=20)
            simulation.is_render = True
            simulation.simulate_episodes(n_episodes=3)
            simulation.is_render = False

        simulation.terminate()


if __name__ == '__main__':
    run_agent()
