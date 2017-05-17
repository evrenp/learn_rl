import pytest
import tensorflow as tf
from gym.envs import make
from agents.deep_q_learning_agent import DeepQLearningAgent
from simulation.simulation import Simulation


def test_deep_q_learning_agent():
    session = tf.Session(graph=tf.get_default_graph())
    env = make("Breakout-v0")
    env._max_episode_steps = 6
    agent = DeepQLearningAgent(
        session=session,
        env=env,
        epsilon=0.5,
        gamma=0.99,
        n_samples_replay_memory=4,
        n_batch_samples=2,
        copy_interval=4,
        logger_level='DEBUG'
    )

    with agent.session:
        agent.tf_init_at_session_start()
        simulation = Simulation(env=env, agent=agent, logger_level='DEBUG')
        simulation.simulate_episodes(n_episodes=2)
        simulation.terminate()
