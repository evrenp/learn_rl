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


def test_q_learning_agent_timeout():
    agent = SarsaMaxAgent(env=env, max_n_steps=4)
    simulation = Simulation(env=env, agent=agent, max_n_steps=5)
    simulation.simulate_episodes(n_episodes=2)
    simulation.terminate()


def test_monte_carlo_agent_timeout():
    agent = MonteCarloAgent(env=env, max_n_steps=5)
    simulation = Simulation(env=env, agent=agent, max_n_steps=5)
    simulation.simulate_episodes(n_episodes=2)
    simulation.terminate()


def test_sarsa_lambda_agent_timeout():
    agent = SarsaLambdaAgent(env=env, max_n_steps=5)
    simulation = Simulation(env=env, agent=agent, max_n_steps=8)
    simulation.simulate_episodes(n_episodes=2)
    simulation.terminate()
