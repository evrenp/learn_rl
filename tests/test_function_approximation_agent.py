import pytest

from gym import make
from simulation.simulation import Simulation
from agents.function_approximation_agents import SarsaMaxFunctionApproximationAgent

def test_function_approximation_agent():
    env = make('MountainCar-v0')
    env._max_episode_steps = 100
    agent = SarsaMaxFunctionApproximationAgent(env=env, epsilon=0.5, gamma=1.0)
    simulation = Simulation(env=env, agent=agent, logger_level='INFO', is_render=False, max_n_steps=5000)
    simulation.simulate_episodes(n_episodes=2)
    simulation.terminate()