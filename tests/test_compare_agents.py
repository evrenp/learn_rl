import pytest
from gym import make

from agents.discrete_agents import SarsaMaxAgent, MonteCarloAgent, SarsaLambdaAgent
from agents.random_agent import RandomAgent
from simulation.simulation import compare_agents


def test_compare_agents():
    env = make('FrozenLake-v0')

    constructor_kwargs_list = [
        (RandomAgent, dict(env=env)),
        (MonteCarloAgent, dict(env=env, epsilon=0.1, gamma=0.8)),
        (SarsaMaxAgent, dict(env=env, epsilon=0.2, gamma=0.8)),
        (SarsaLambdaAgent, dict(env=env, epsilon=0.2, gamma=0.8)),
    ]
    _, _ = compare_agents(env=env, constructor_kwargs_list=constructor_kwargs_list, n_iter=2, n_episodes=10,
                          n_jobs=-1)
