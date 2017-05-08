import matplotlib.pyplot as plt
from gym import make
from gym.envs import registry, register

from agents.discrete_agents import SarsaMaxAgent, MonteCarloAgent, SarsaLambdaAgent
from agents.random_agent import RandomAgent
from simulation.simulation import compare_agents

import numpy as np
import random
np.random.seed(1)
random.seed(1)

if 'GridEnv-v0' not in registry.env_specs:
    register(
        id='GridEnv-v0',
        entry_point='environments.grid_env:GridEnv',
        kwargs={'width': 4, 'height': 3},
    )
env = make('GridEnv-v0')

constructor_kwargs_list = [
    (RandomAgent, dict(env=env)),
    (MonteCarloAgent, dict(env=env, epsilon=0.1, gamma=0.8)),
    (SarsaMaxAgent, dict(env=env, epsilon=0.2, gamma=0.8)),
    (SarsaLambdaAgent, dict(env=env, epsilon=0.2, gamma=0.8)),
]
df, fig = compare_agents(env=env, constructor_kwargs_list=constructor_kwargs_list, n_iter=1, n_episodes=300,
                         n_jobs=-1)
plt.show()

# TODO: main with argparse