from gym import make
from simulation.simulation import Simulation
from agents.sarsa_max_function_approximation_sklearn import SarsaMaxFunctionApproximationAgentSklearn


# import numpy as np
# import random
# np.random.seed(1)
# random.seed(1)


if __name__ == '__main__':

    env = make('MountainCar-v0')
    env._max_episode_steps = 400
    agent = SarsaMaxFunctionApproximationAgentSklearn(env=env, epsilon=0.5, gamma=1.0, save_monitor=True)
    simulation = Simulation(env=env, agent=agent, logger_level='INFO', is_render=False, max_n_steps=5000)
    # simulation.simulate_episodes(n_episodes=1)
    for i in range(100):
        simulation.simulate_episodes(n_episodes=10)
        simulation.is_render = True
        simulation.simulate_episodes(n_episodes=1)
        simulation.is_render = False

    simulation.terminate()