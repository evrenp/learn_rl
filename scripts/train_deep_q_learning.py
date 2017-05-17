import argparse
import os
import tensorflow as tf
from gym.envs import make
from agents.deep_q_learning_agent import DeepQLearningAgent
from simulation.simulation import Simulation

DATA_PATH = os.path.join(os.getenv('HOME'), 'reinforcement_learning')
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, help="id of running training process", default="default")
    parser.add_argument("--n_episodes", type=int, default=2)
    args = parser.parse_args()

    session = tf.Session(graph=tf.get_default_graph())
    env = make("Breakout-v0")
    # env._max_episode_steps = 10
    agent = DeepQLearningAgent(
        session=session,
        env=env,
        epsilon=0.2,
        gamma=0.99,
        n_samples_replay_memory=2000,
        n_batch_samples=32,
        copy_interval=1000,
        logger_level='DEBUG'
    )

    with agent.session:
        agent.tf_init_at_session_start()
        simulation = Simulation(env=env, agent=agent, is_render=False, logger_level='DEBUG', save_summary=True, episode_interval_save_summary=5)
        simulation.simulate_episodes(n_episodes=args.n_episodes)
        simulation.terminate()

if __name__ == '__main__':
    main()