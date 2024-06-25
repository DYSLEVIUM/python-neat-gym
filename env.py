# https://gymnasium.farama.org/content/basic_usage/

import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
# env = gym.make("LunarLander-v2", render_mode="human")

observation, info = env.reset(seed=42)

print("observation: ", observation)
print("action space: ", env.action_space)

for _ in range(10000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

    env.render()

env.close()
