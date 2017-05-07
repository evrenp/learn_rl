import logging
import sys
from itertools import product

import numpy as np
import pandas as pd

from gym import undo_logger_setup
from joblib import Parallel, delayed
from matplotlib import pyplot as plt


class Simulation(object):
    def __init__(
            self,
            env,
            agent,
            logger_level='INFO',
            max_n_steps=10000,
            is_render=False,
    ):
        """Init

        Args:
            env (gym.Env): environment
            agent (ai_gym_experiments.agents.base_agent.BaseAgent): agent
            logger_level (str): logger level
            max_n_steps (int): max number of steps
            is_render (bool): whether or not to render
        """
        self.logger = self._get_logger(level=logger_level)

        # dynamic variables
        self.env = env
        self.agent = agent

        self.max_n_steps = max_n_steps
        # overwrite agent.max_n_steps
        if hasattr(agent, 'max_n_steps') and (self.agent.max_n_steps < self.max_n_steps):
            self.agent.max_n_steps = self.max_n_steps
            self.logger.warning('agent.max_n_steps has been overwritten by simulation.')

        self.is_render = is_render


    @staticmethod
    def _get_logger(level):
        assert level in ['CRITICAL', 'ERROR', 'WARN', 'WARNING', 'INFO', 'DEBUG', 'NOTSET']
        undo_logger_setup()
        logger = logging.getLogger()
        formatter = logging.Formatter('%(message)s')
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
        return logger

    def simulate_single_episode(self):

        # episode starts with observation and reward at t=0
        observation = self.env.reset()
        reward = None
        done = False

        self.agent.reset_before_new_episode()

        for t in range(self.max_n_steps - 1):
            self.agent.set_time(t=t)

            if self.is_render:
                self.env.render()

            # observation at t, reward for moving into t, action at t for moving into t+1
            action = self.agent.act(observation, reward, done)

            # observation at t+1, reward for moving into t+1
            observation, reward, done, info = self.env.step(action)

            # timeout
            if t == self.max_n_steps - 2:
                done = True
                self.logger.warning('max_n_steps={} has been reached.'.format(self.max_n_steps))

            if done:
                # final info (observation, reward, done) are given to the agent via act method
                self.agent.set_time(t=t + 1)
                _ = self.agent.act(observation, reward, done)
                break

        self.logger.debug('\nobservation_path:\n{}'.format(self.agent.observation_path.round(2)))
        self.logger.debug('\naction_path:\n{}'.format(self.agent.action_path.round(2)))
        self.logger.debug('\nreward_path:\n{}'.format(self.agent.reward_path.round(2)))

    def simulate_episodes(self, n_episodes=2):
        for idx_episode in range(n_episodes):
            self.logger.debug('\nEpisode {}:'.format(idx_episode))
            self.simulate_single_episode()

    def terminate(self):
        self.env.close()


def _parallel_function(env, iter_idx, n_episodes, constructor, kwargs):
    agent = constructor(**kwargs)
    agent_id = agent.get_id()
    simulation = Simulation(env=env, agent=agent)

    index = pd.MultiIndex.from_product([[agent_id], [iter_idx], range(n_episodes)],
                                       names=['agent_id', 'iter_idx', 'episode_idx'])
    df = pd.DataFrame(columns=['steps_per_episode', 'reward_per_episode', 'reward_per_step'], index=index, dtype=float)

    for episode_idx in range(n_episodes):
        simulation.simulate_single_episode()
        df.loc[(agent_id, iter_idx, episode_idx), 'steps_per_episode'] = len(simulation.agent.reward_path)
        df.loc[(agent_id, iter_idx, episode_idx), 'reward_per_episode'] = np.nansum(simulation.agent.reward_path)
        df.loc[(agent_id, iter_idx, episode_idx), 'reward_per_step'] = np.nanmean(simulation.agent.reward_path)
    simulation.terminate()
    return df


def compare_agents(env, constructor_kwargs_list, n_iter=5, n_episodes=100, n_jobs=1):
    """Compare different agent in one environment

    Args:
        env (gym.Env): environment
        constructor_kwargs_list (list): list of tuples (agent constructor, kwargs of constructor)
        n_iter (int): number of iterations/repetitions of the experiment
        n_episodes (int): number of episodes
        n_jobs (int): number of parallel jobs

    Returns:
        df (pd.DataFrame): results data frame
        fig (plt.figure.Figure): results figure

    Example:
        >> env = make('FrozenLake-v0')
        >> constructor_kwargs_list = [(RandomAgent, dict(env=env)), (SarsaMaxAgent, dict(env=env, epsilon=0.1, gamma=0.8))]
        >> df, fig = compare_agents(env=env, constructor_kwargs_list=constructor_kwargs_list, n_iter=10, n_episodes=1000, n_jobs=-1)
        >> plt.show()
    """

    args = product(range(n_iter), constructor_kwargs_list)
    results = Parallel(n_jobs=n_jobs)(
        delayed(_parallel_function)(env, iter_idx, n_episodes, constructor, kwargs) for iter_idx, (constructor, kwargs)
        in args)
    df = pd.concat(results, axis=0, ignore_index=False)

    # mean over iterations
    df = df.groupby(level=['agent_id', 'episode_idx']).mean()

    # make plot
    fig, axes = plt.subplots(3, 1)
    for ax_idx, col in enumerate(['steps_per_episode', 'reward_per_episode', 'reward_per_step']):
        df.reset_index().pivot('episode_idx', 'agent_id', col).rolling(center=True,
                                                                       window=int(n_episodes * 0.1) + 1).mean().plot(
            ax=axes[ax_idx])
        axes[ax_idx].set(ylabel=col)
    fig.suptitle(env.__class__.__name__, fontsize=12)
    return df, fig
