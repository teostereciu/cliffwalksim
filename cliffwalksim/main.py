import gymnasium as gym

from agents.agentfactory import AgentFactory


def env_interaction(env_str: str, agent_type: str, num_episodes: int = 500) -> None:
    env = gym.make(env_str, render_mode='human')
    obs, info = env.reset()
    agent = AgentFactory.create_agent(agent_type, env=env)

    while True:
        old_obs = obs
        action = agent.policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        agent.update((old_obs, action, reward, obs))

        if terminated or truncated:
            num_episodes -= 1
            # Episode ended.
            obs, info = env.reset()

        if num_episodes == 0:
            break

    env.close()


if __name__ == "__main__":
    env_interaction("CliffWalking-v0", 'RANDOM')



