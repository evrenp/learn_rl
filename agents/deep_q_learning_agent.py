"""
Checkout: https://github.com/dennybritz/reinforcement-learning/blob/master/DQN/Deep%20Q%20Learning%20Solution.ipynb
"""
from gym.wrappers import Monitor
import itertools
import numpy as np
import os
import random
import sys
import tensorflow as tf
from collections import deque, namedtuple
from gym import make
from simulation.simulation import Simulation
from agents.random_agent import RandomAgent


# Atari Actions: 0 (noop), 1 (fire), 2 (left) and 3 (right) are valid actions
VALID_ACTIONS = [0, 1, 2, 3]


class StateProcessor():
    """
    Processes a raw Atari iamges. Resizes it and converts it to grayscale.
    """
    def __init__(self):
        # Build the Tensorflow graph
        with tf.variable_scope("state_processor"):
            self.input_state = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
            self.output = tf.image.rgb_to_grayscale(self.input_state)
            self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
            self.output = tf.image.resize_images(
                self.output, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.output = tf.squeeze(self.output)

    def process(self, sess, state):
        """
        Args:
            sess: A Tensorflow session object
            state: A [210, 160, 3] Atari RGB State

        Returns:
            A processed [84, 84, 1] state representing grayscale values.
        """
        return sess.run(self.output, { self.input_state: state })



if __name__ == '__main__':

    env = make("Breakout-v0")
    # print(env.action_space.sample())
    # print('shape of observations:', env.observation_space.sample().shape)
    # env._max_episode_steps = 400
    agent = RandomAgent(env=env)
    simulation = Simulation(env=env, agent=agent, logger_level='INFO', is_render=True, max_n_steps=5000)
    simulation.simulate_episodes(n_episodes=2)
    simulation.terminate()

    # input = tf.placeholder(shape=[3], dtype=tf.uint8)
    # output = 10 * input
    #
    # with tf.Session() as session:
    #     output_value = session.run(output, {input: np.ones(3)})
    #
    # print(output_value)