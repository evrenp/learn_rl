import pytest
from gym import make
from gym.envs import register, registry
from agents.random_agent import RandomAgent
from simulation.simulation import Simulation


def test_random_agent_in_environments():
    if 'GridEnv-v0' not in registry.env_specs:
        register(
            id='SimpleGridEnv-v0',
            entry_point='environments.grid_env:GridEnv',
            kwargs={'width': 3, 'height': 2},
        )

    env_ids = [
        'MountainCar-v0',
        'MountainCarContinuous-v0',
        'CartPole-v0',
        'AirRaid-ram-v0',
        'GridEnv-v0'
    ]

    for env_id in env_ids:
        env = make(env_id)
        agent = RandomAgent(env=env)
        simulation = Simulation(env=env, agent=agent)
        simulation.simulate_episodes(n_episodes=3)
        simulation.terminate()