import gym
import numpy as np
from gym import wrappers

env = gym.make('CartPole-v0')

done = False
bestLength = 0
episode_length = []

best_weights = np.zeros(4)
cnt = 0
#for i in range(100):
   # new_weights = np.random.uniform(-1, 1, 4)
env.reset()
while not done:
    env.render()
    cnt += 1

    action = env.action_space.sample()

    observation, reward, done, _ = env.step(action)

    if done:
        break

print('game lasted: ', cnt, 'moves')

