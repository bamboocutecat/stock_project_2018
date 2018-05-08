import gym
import time


env = gym.make('CartPole-v0')
env.reset()
env.render()
time.sleep(10)
env.close()