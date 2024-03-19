import gymnasium as gym

from agents.agentfactory import AgentFactory
from cliffwalksim.util.metricstracker import MetricsTracker


def env_interaction(env_str: str, agent_type: str, num_episodes: int = 5, tracker: MetricsTracker = None) -> None:
    env = gym.make(env_str, render_mode='None')
    obs, info = env.reset()
    agent = AgentFactory.create_agent(agent_type, env=env)
    
    if tracker is None:
        tracker = MetricsTracker()
    episode_reward = 0

    while True:
        old_obs = obs
        action = agent.policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        agent.update((old_obs, action, reward, obs))

        if terminated or truncated:
            num_episodes -= 1
            # Episode ended.
            tracker.record_return(agent_type, episode_reward)
            episode_reward = 0
            obs, info = env.reset()

        if num_episodes == 0:
            break

    env.close()
    return tracker


if __name__ == "__main__":
    agent_types = ['SARSA', 'Q_LEARNING', 'DOUBLE_Q_LEARNING', 'RANDOM']
    all_trackers = {}
    all_rewards = []

    for agent_type in agent_types:
        old_tracker = None
        for i in range(2):
            new_tracker = env_interaction("CliffWalking-v0", agent_type, tracker=old_tracker)
            old_tracker = new_tracker
        all_trackers[agent_type] = old_tracker
       

    for agent_type, tracker in all_trackers.items():
        tracker.plot()



