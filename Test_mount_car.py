import gym
env = gym.make("MountainCar-v0")
obs = env.reset()
print(obs)
print(env.observation_space,env.action_space)
print(env.observation_space.shape[0])
