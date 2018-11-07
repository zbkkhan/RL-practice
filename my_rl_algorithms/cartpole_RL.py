import numpy as np
import gym

#Bellman equation


#Steps:
# 1. Setup Q table
# 2. Update Q table

epoch = 1.0
MIN_EPOCH = 0.2
MAX_EPOCH = 1.0
LR = 0.2
GAMMA = 0.3

MAX_EPISODES = 10000
N_STATES = 48 #For Cart pole there are 48 degrees the pole can be (from -24 to 24 degrees)
ACTION_SPACE = 2 #You can either go left or right

def obs_to_state(env, obs):
    env_high = env.observation_space.high
    env_low = env.observation_space.low
    state = int(obs[2]/((env_high[2] - env_low[2])/N_STATES))# use the pole angle as state
    return state




def build_table(n_states = N_STATES, action_space = ACTION_SPACE):
    table = np.zeros((n_states, action_space))
    return table

def run_training_episode(q_table, env, S ):
    env.render()
    done = False
    i = 1
    while not (done): # terminate if 200 time steps have been achieved
        #choose an action from either a random action or from the Q table (exploration vs exploitation)
        if((np.random.uniform(0, 1) > epoch)):
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[S, :])

        #take the action in the environment
        obs, reward, done, _ = env.step(action)
        S_ = obs_to_state(env, obs)
        q_table[S, action] += LR * (reward +((GAMMA) * max(q_table[S_, :]) - q_table[S, action]))
        i += 1
        S = S_
    return q_table, i



if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    print(env.observation_space.high)
    q_table = build_table()
    env.seed(0)
    total_steps = 0

    for i in range(MAX_EPISODES):
        #Run an episode
        obs = env.reset()
        S = obs_to_state(env, obs)
        q_table, steps = run_training_episode(q_table, env, S)
        total_steps += steps
        #update epoch to allow for more exploitation than exploration as episodes continue
        epoch = -((MAX_EPOCH - MIN_EPOCH)/MAX_EPISODES) * i + MAX_EPOCH

        if((i % 1000) == 0):
            print("Average survived steps: ",(total_steps/1000))
            total_steps = 0

    print(q_table)


    #for _ in range(1000):
     #   env.render()
      #  env.step(env.action_space.sample()) # take a random action
