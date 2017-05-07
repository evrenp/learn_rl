import pytest
from gym import make
from gym.envs import register, registry
from agents.discrete_agents import SarsaMaxAgent, MonteCarloAgent, SarsaLambdaAgent
from simulation.simulation import Simulation

if 'GridEnv-v0' not in registry.env_specs:
    register(
        id='GridEnv-v0',
        entry_point='environments.grid_env:GridEnv',
        kwargs={'width': 3, 'height': 2},
    )
env = make('GridEnv-v0')


def test_q_learning_agent():
    agent = SarsaMaxAgent(env=env)
    simulation = Simulation(env=env, agent=agent)
    simulation.simulate_episodes(n_episodes=2)
    simulation.terminate()


def test_monte_carlo_agent():
    agent = MonteCarloAgent(env=env)
    simulation = Simulation(env=env, agent=agent)
    simulation.simulate_episodes(n_episodes=2)
    simulation.terminate()


def test_sarsa_lambda_agent():
    agent = SarsaLambdaAgent(env=env)
    simulation = Simulation(env=env, agent=agent)
    simulation.simulate_episodes(n_episodes=2)
    simulation.terminate()
