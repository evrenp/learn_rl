import pytest

from gym import make
from simulation.simulation import Simulation
from agents.sarsa_max_function_approximation_sklearn import SarsaMaxFunctionApproximationAgentSklearn


def test_sarsa_max_function_approximation_sklearn():
    env = make('MountainCar-v0')
    env._max_episode_steps = 100
    agent = SarsaMaxFunctionApproximationAgentSklearn(env=env, epsilon=0.5, gamma=1.0)
    simulation = Simulation(env=env, agent=agent, is_render=False)
    simulation.simulate_episodes(n_episodes=2)
    simulation.terminate()
