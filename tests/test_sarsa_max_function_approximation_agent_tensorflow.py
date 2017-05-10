import pytest

from gym import make
from simulation.simulation import Simulation
from agents.sarsa_max_function_approximation_tensorflow import SarsaMaxFunctionApproximationAgentTF


def test_sarsa_max_function_approximation_tensorflow():
    env = make('MountainCar-v0')
    env._max_episode_steps = 100
    agent = SarsaMaxFunctionApproximationAgentTF(env=env, epsilon=0.5, gamma=1.0)
    with agent.session:
        agent.tf_init_at_session_start()
        simulation = Simulation(env=env, agent=agent, is_render=False)
        simulation.simulate_episodes(n_episodes=2)
        simulation.terminate()
