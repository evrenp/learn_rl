import matplotlib.pyplot as plt

from gym import make
from agents.random_agent import RandomAgent
from agents.sarsa_max_function_approximation_sklearn import SarsaMaxFunctionApproximationAgentSklearn
from simulation.simulation import compare_agents

env = make('MountainCar-v0')
env._max_episode_steps = 400

constructor_kwargs_list = [
    (RandomAgent, dict(env=env)),
    (SarsaMaxFunctionApproximationAgentSklearn, dict(env=env, epsilon=0.5, gamma=1.0, feature_creation='scaling')),
    (SarsaMaxFunctionApproximationAgentSklearn, dict(env=env, epsilon=0.5, gamma=1.0, feature_creation='scaling_and_rbf')),
]
df, fig = compare_agents(env=env, constructor_kwargs_list=constructor_kwargs_list, n_iter=1, n_episodes=100,
                         n_jobs=-1)
plt.show()

