import gym
from gym.spaces import Discrete, Tuple, Dict
import random
import numpy as np

class CustomCorrelatedActionsEnv(gym.Env):
    """
    Simple env in which the policy has to emit a tuple of equal actions.
    In each step, the agent observes a random number (0 or 1) and has to choose
    two actions a1 and a2.
    It gets +5 reward for matching a1 to the random obs and +5 for matching a2
    to a1. I.e., +10 at most per step.
    One way to effectively learn this is through correlated action
    distributions, e.g., in examples/autoregressive_action_dist.py
    There are 20 steps. Hence, the best score would be ~200 reward.
    """

    def __init__(self, _):
        self.observation_space = Discrete(2)
        #self.action_space = Tuple([Discrete(2), Discrete(2)])

        #"""
        #observation_space = {
        #    'obs': Discrete(2), #values between [0,2), i.e. EXCLUSIVE 2
        #    'a_1': Discrete(2)
        #}

        #self.observation_space= gym.spaces.Dict(observation_space)

        action_spaces = {
            'a_1': Discrete(2),
            'a_2': Discrete(2)
        }
        self.action_space = gym.spaces.Dict(action_spaces)

        self.last_observation = None
        #""" or None

    def reset(self):
        self.t = 0
        self.last_observation = random.choice([0, 1])
        #print(f"LAST OBS: {self.last_observation}")
        return self.observation_wrapper(self.last_observation)

    def step(self, action):
        #print("TEST STEP")
        #print(action)

        self.t += 1
        #a1, a2 = action

        a1 = action.get("a_1")
        a2 = action.get("a_2")

        reward = 0
        # Encourage correlation between most recent observation and a1.
        if a1 == self.last_observation:
            reward += 5
        # Encourage correlation between a1 and a2.
        if a1 == a2:
            reward += 5
        done = self.t > 20
        self.last_observation = random.choice([0, 1])
        return self.observation_wrapper(self.last_observation), reward, done, {}

    def observation_wrapper(self, value):
        return value
        #return  {
        #    'obs': value,
        #    'a_1': 0 # dummy_value
        #}



class CustomCorrelatedActionsDirichletEnv(gym.Env):
    """
    Simple env in which the policy has to emit a tuple of equal actions.
    In each step, the agent observes a random number (0 or 1) and has to choose
    two actions a1 and a2.
    It gets +5 reward for matching a1 to the random obs and +5 for matching a2
    to a1. I.e., +10 at most per step.
    One way to effectively learn this is through correlated action
    distributions, e.g., in examples/autoregressive_action_dist.py
    There are 20 steps. Hence, the best score would be ~200 reward.
    """

    def __init__(self, _):

        print("RUN CUSTOM ENVIRONMENTGYM")
        amount_observations = 2
        self.amount_observations = amount_observations
        #gym.spaces.Box(low=-np.inf, high=np.inf, shape=(amount_observations,), dtype=np.float)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(amount_observations,), dtype=np.float) #Discrete(2)
        #self.action_space = Tuple([Discrete(2), Discrete(2)])

        #"""
        #observation_space = {
        #    'obs': Discrete(2), #values between [0,2), i.e. EXCLUSIVE 2
        #    'a_1': Discrete(2)
        #}

        #self.observation_space= gym.spaces.Dict(observation_space)

        action_spaces = {
            'a_1': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(amount_observations,), dtype=np.float),#Discrete(2),
            'a_2': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(amount_observations,), dtype=np.float),#Discrete(2)
        }
        self.action_space = gym.spaces.Dict(action_spaces)

        self.last_observation = None
        #""" or None

    def sample_from_dirichlet(self):
        dirich_param = np.ones(self.amount_observations)
        # print(dirich_param)
        dir_samples = np.random.dirichlet(dirich_param, size=1).flatten()#self.amount_observations)

        return dir_samples


    def reset(self):
        self.t = 0
        self.last_observation = self.sample_from_dirichlet()#random.choice([0, 1])
        #print(f"LAST OBS: {self.last_observation}")
        return self.observation_wrapper(self.last_observation)

    def observation_wrapper(self, value):
        return value

    def step(self, action):

        self.t += 1
        #a1, a2 = action

        a1 = action.get("a_1")
        a2 = action.get("a_2")

        reward = 0
        # Encourage correlation between most recent observation and a1.
        reward -= np.linalg.norm(self.last_observation - a1, ord=2)
        #if a1 == self.last_observation:
        #    reward += 5
        # Encourage correlation between a1 and a2.
        reward -= np.linalg.norm(a1-a2, ord=2)

        #if a1 == a2:
        #    reward += 5
        done = self.t > 20
        self.last_observation = self.sample_from_dirichlet()#random.choice([0, 1])
        #print("ENV")
        #print(self.last_observation)
        return self.observation_wrapper(self.last_observation), reward, done, {}