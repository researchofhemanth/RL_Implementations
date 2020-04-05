import gym
import mujoco_py
from gym.envs import mujoco
env = gym.make("HalfCheetah-v2")
env.reset()
print(env.observation_space,env.action_space)
