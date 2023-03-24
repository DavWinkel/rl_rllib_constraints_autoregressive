import gym
import numpy as np
import random
from typing import Tuple, List, Dict
from ray.rllib.utils.spaces.simplex import Simplex
from copy import deepcopy
from ray.rllib.env.env_context import EnvContext

from ray.rllib.utils.numpy import one_hot as np_one_hot
from helper_functions import calculate_portfolio_variance_from_action, convert_conditional_minkowski_encoding_to_action
import re
import math

#from helper_functions import reverse_one_hot_np
from financial_markets_gym.envs.financial_markets_env import FinancialMarketsEnv
#from helper_functions import calculate_portfolio_variance_from_action
import pandas as pd
import inspect
import torch

def extract_environment_class_from_config(policy_config):

    if "wrapped-financial-markets-env-v0" in policy_config.get("env"):
        return BasicWrapperFinancialEnvironmentEvaluationable
    elif "wrapped-financial-markets-env-short-selling-v0" in policy_config.get("env"):
        return BasicWrapperFinancialEnvironmentShortSelling

#to install custom environments use:
# pip install -e custom-gym
# pip install -e financial-markets-gym

class BasicWrapperFinancialMarkets(gym.Wrapper):

    def __init__(self, env, config: EnvContext):
        super().__init__(env)

        self.env = env
        self.terminal_time_step = 12
        self.amount_hidden_states = 2

        #card_discrete_action_space = self.action_space.n

        #print("Finance")
        #print(self.action_space.shape)
        # overwriting the action_space (which was discrete to a "meta view", i.e. we return a probablistic action
        # self.action_space = ray.rllib.models.extra_spaces.Simplex(shape=(card_discrete_action_space, ))
        self.action_space = Simplex(shape=self.action_space.shape)

        # Overwriting the previous observation space with the observation space of the wrapped one hot encoded
        # observation space
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(
            self.observation_space.shape[0]-1+self.amount_hidden_states,), dtype=np.float)

    def step(self, action) -> Tuple[np.ndarray, np.ndarray, bool, dict]:

        next_state, reward, done, info = self.env.step(action)

        return self.observation_wrapper(next_state), reward, done, info

    def reset(self):

        state = self.env.reset()

        return self.observation_wrapper(state)

    def observation_wrapper(self, state):

        predicted_hidden_state = state[0].astype(int)
        np_one_hot_predicted_hidden_state = np_one_hot(predicted_hidden_state, depth=self.amount_hidden_states)

        return np.concatenate((np_one_hot_predicted_hidden_state, state[1:]))



def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()

def step_test(env, a):
    transitions = env.P[env.s][a]
    i = categorical_sample([t[0] for t in transitions], env.np_random)
    p, s, r, d = transitions[i]
    env.s = s
    env.lastaction = a
    return (int(s), r, d, {"prob": p})

#This is NOT a subclass of the env, but rather a WRAPPER
class BasicWrapperDiscreteEnv(gym.Wrapper):
    """
    This class is meant as a general wrapper around all discrete environments
    """
    def __init__(self, env, config: EnvContext):

        super().__init__(env)

        self.env = env

        card_discrete_action_space = self.action_space.n

        #overwriting the action_space (which was discrete to a "meta view", i.e. we return a probablistic action
        #self.action_space = ray.rllib.models.extra_spaces.Simplex(shape=(card_discrete_action_space, ))
        self.action_space = Simplex(shape=(card_discrete_action_space, ))

        self.is_one_period_evaluation = False
        self.starting_state_to_evaluate = None
        self.meta_action_to_evaluate = None

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, dict]:

        if self.is_one_period_evaluation:
            if self.meta_action_to_evaluate is None:
                raise ValueError(f'meta_action_to_evaluate is set to None')
            action = self.meta_action_to_evaluate

        action_64bit_norm = action.astype(np.float64) / (np.sum(action.astype(np.float64)))
        discrete_action = np.argmax(np.random.multinomial(1, pvals=action_64bit_norm, size=1))
        #if self.is_one_period_evaluation:
            #print("Initial:")
            #print(self.env.super(super(FrozenLakeEnv, self)).s)
            #discrete_action = 2
            #print(self.env.s)
            #print(self.env.P)
            #print(self.env.P[self.env.s][discrete_action])
            #print("TEST")
            #print(step_test(self.env, discrete_action))
        #
            #print(discrete_action)
        #if self.is_one_period_evaluation:
        #    print("##")
        #    print(discrete_action)
        #    print(self.unwrapped.P[self.unwrapped.s][discrete_action])
        next_state, reward, done, info = self.env.step(discrete_action)

        if self.is_one_period_evaluation:
            done = True
            #print(f'Sampled next state: {next_state}')
            #print(f'Reward: {reward}')

        return next_state, reward, done, info


    def reset(self):
        #self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
        if self.is_one_period_evaluation:

            self.env.reset()
            # in case of frozen lake
            # potentially we have to do some state conversion
            if self.starting_state_to_evaluate is None:
                raise ValueError(f'evaluation_state is set to None')
            #print(f"RESET: {self.env.s}")

            #self.unwrapped.s = self.starting_state_to_evaluate #We have to work here on the

            # #INNER LAYER using self.unwrapped and not self.env.s -> At least for frozen lake
            # see https://stackoverflow.com/questions/53836136/why-unwrap-an-openai-gym
            #return self.unwrapped.s
            return self.set_and_return_starting_state_to_evaluate(self.starting_state_to_evaluate)
        else:
            return self.env.reset()

    def set_and_return_starting_state_to_evaluate(self, starting_state_to_evaluate):
        self.unwrapped.s = self.starting_state_to_evaluate  # We have to work here on the
        # #INNER LAYER using self.unwrapped and not self.env.s -> At least for frozen lake
        # see https://stackoverflow.com/questions/53836136/why-unwrap-an-openai-gym
        return self.unwrapped.s

    def set_one_period_evaluation_mode(self, is_one_period_evaluation):
        self.is_one_period_evaluation = is_one_period_evaluation

    def set_starting_state_to_evaluate(self, starting_state_to_evaluate):
        self.starting_state_to_evaluate = starting_state_to_evaluate

    def set_meta_action_to_evaluate(self, meta_action_to_evaluate):
        self.meta_action_to_evaluate = meta_action_to_evaluate

    def get_unwrapped_environment_observation_space(self):
        return self.env.observation_space #this returns the initial observation_space before the wrapping

class BasicWrapperFrozenLake(BasicWrapperDiscreteEnv):

    def set_and_return_starting_state_to_evaluate(self, starting_state_to_evaluate):
        self.unwrapped.s = self.starting_state_to_evaluate  # We have to work here on the
        # #INNER LAYER using self.unwrapped and not self.env.s -> At least for frozen lake
        # see https://stackoverflow.com/questions/53836136/why-unwrap-an-openai-gym
        return self.unwrapped.s

class BasicWrapperCustomGridworld(BasicWrapperDiscreteEnv):

    def set_and_return_starting_state_to_evaluate(self, starting_state_to_evaluate):
        self.unwrapped.current_state = self.starting_state_to_evaluate  # We have to work here on the
        # #INNER LAYER using self.unwrapped and not self.env.s -> At least for frozen lake
        # see https://stackoverflow.com/questions/53836136/why-unwrap-an-openai-gym
        return self.unwrapped.current_state

class BasicWrapperLunarLander(BasicWrapperDiscreteEnv):

    def set_and_return_starting_state_to_evaluate(self, starting_state_to_evaluate):
        """
        s[0] is the horizontal coordinate
                  s[1] is the vertical coordinate
                  s[2] is the horizontal speed
                  s[3] is the vertical speed
                  s[4] is the angle
                  s[5] is the angular speed
                  s[6] 1 if first leg has contact, else 0
                  s[7] 1 if second leg has contact, else 0
        :param starting_state_to_evaluate:
        :return:
        """

        self.unwrapped.current_state = self.starting_state_to_evaluate  # We have to work here on the
        # #INNER LAYER using self.unwrapped and not self.env.s -> At least for frozen lake
        # see https://stackoverflow.com/questions/53836136/why-unwrap-an-openai-gym
        return self.unwrapped.current_state


class BasicWrapperFinancialEnvironmentEvaluationable(gym.Wrapper):

    def __init__(self, env, config: EnvContext, apply_softmax_to_actions=False,
                 allow_simplex_action_space_definition=True):
        super().__init__(env)

        self.env = env

        if allow_simplex_action_space_definition:
            self.action_space = Simplex(shape=self.action_space.shape)

        print("WE ARE IN A NEW ENVIRONMENT")

        if config.get("include_risk_penalty_in_state") is not None:
            self.include_risk_penalty_in_state = config.get("include_risk_penalty_in_state")
        else:
            if ("risk_penalty_factor_lower_bound" in config) & ("risk_penalty_factor_upper_bound" in config) &\
                (config["risk_penalty_factor_lower_bound"] is not None) & (config["risk_penalty_factor_upper_bound"] is not None):
                print("WARNING: Please state expliciticly the 'include_risk_penalty_in_state' in the env config")
                self.include_risk_penalty_in_state = True
            else:
                raise ValueError(f"Upper and lower bound need to be set")

        if ("risk_penalty_factor_lower_bound" in config) & ("risk_penalty_factor_upper_bound" in config) &\
                (config["risk_penalty_factor_lower_bound"] is not None) & (config["risk_penalty_factor_upper_bound"] is not None):
            if not self.include_risk_penalty_in_state:
                raise ValueError(f"When using upper and lower bound the inclusion of risk penaltiy in the state is necessary")
            self.randomness_for_risk_penalty = True
            self.risk_penalty_lower_bound = config.get("risk_penalty_factor_lower_bound")
            self.risk_penalty_upper_bound = config.get("risk_penalty_factor_upper_bound")
            self.risk_penalty_current = None
        elif ("risk_penalty_factor" in config):
            self.risk_penalty_current = config.get("risk_penalty_factor")
            self.randomness_for_risk_penalty = False
        else:
            self.randomness_for_risk_penalty = False

        if self.include_risk_penalty_in_state:
            # Overwriting the previous observation space with the observation space of the wrapped one hot encoded
            # observation space
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(
                self.observation_space.shape[0]+1,), dtype=np.float)


        if "apply_softmax_to_actions" in config:
            apply_softmax_to_actions = config.get("apply_softmax_to_actions")

        self.apply_softmax_to_actions = apply_softmax_to_actions
        if self.apply_softmax_to_actions:
            # since we are using LOGIT we DO NOT NEED TO BOUND ANY MORE ->
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=self.action_space.shape,
                                                    dtype=np.float)

        if self.unwrapped.include_unobservable_market_state_information_for_evaluation:
            # Overwriting the previous observation space with the observation space of the wrapped one hot encoded
            # observation space
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(
                self.observation_space.shape[0]-1,), dtype=np.float)

        self.is_one_period_evaluation = False
        self.starting_state_to_evaluate = None
        self.is_helper_sampling_mode = False

        self.action_to_evaluate = None
        self.memory_mode = False
        self.initial_reset_state = None

        self.memory_action = []
        self.memory_full_state = [] # including also hidden information which are not observable for the agent
        self.memory_done = []

        self.memory_logit = []
        import datetime
        dateTimeObj = datetime.datetime.now()
        self.timestampStr = dateTimeObj.strftime("%d-%b-%Y (%H:%M:%S.%f)")

    def set_risk_penalty_current(self, risk_penalty_current):
        self.risk_penalty_current = risk_penalty_current

    def set_randomness_for_risk_penalty(self, randomness_for_risk_penalty):
        self.randomness_for_risk_penalty = randomness_for_risk_penalty

    @staticmethod
    def softmax(x, axis=None):
        x = x - x.max(axis=axis, keepdims=True)
        y = np.exp(x)
        return y / y.sum(axis=axis, keepdims=True)

    def step(self, action) -> Tuple[np.ndarray, np.ndarray, bool, dict]:
        if self.apply_softmax_to_actions:
            action=action*10
            self.memory_logit.append(action)
            action = BasicWrapperFinancialEnvironmentEvaluationable.softmax(action)

        if self.is_one_period_evaluation:
            if self.action_to_evaluate is None:
                raise ValueError(f'meta_action_to_evaluate is set to None')
            action = self.action_to_evaluate

        next_state, reward, done, info = self.env.step(action)

        if self.is_one_period_evaluation:
            done = True

        if self.memory_mode:

            if self.initial_reset_state is not None: # appending the initial state from the reset()
                self.memory_full_state.append(self.initial_reset_state)
                self.initial_reset_state = None # emptying the queue

            if not done:
                self.memory_full_state.append(next_state)  # we have a lag of -1 here
            self.memory_action.append(action)
            self.memory_done.append(done)

        return self.observation_wrapper(next_state), reward, done, info

    def observation_wrapper(self, state):

        if self.unwrapped.include_unobservable_market_state_information_for_evaluation:
            state = state[1:] # cut out the unobservable hidden state so the agent can not see this information
        else:
            state = state

        if self.include_risk_penalty_in_state:
            state = np.concatenate(([self.risk_penalty_current], state))
        else:
            state = state

        return state

    def set_one_period_evaluation_mode(self, is_one_period_evaluation):
        self.is_one_period_evaluation = is_one_period_evaluation

    def set_starting_state_to_evaluate(self, starting_state_to_evaluate):
        self.starting_state_to_evaluate = starting_state_to_evaluate

    def set_action_to_evaluate(self, action_to_evaluate):
        self.action_to_evaluate = action_to_evaluate

    def get_unwrapped_environment_observation_space(self):
        return self.env.observation_space # this returns the initial observation_space before the wrapping

    def get_backtesting_ground_truth_risk_per_action(self, np_action):
        """
        Ground truth here is the historical observed risk per month
        :param np_action:
        :return:
        """

        if self.apply_softmax_to_actions:
            action=np_action*10
            self.memory_logit.append(action)
            if np_action.ndim==2:
                np_action = BasicWrapperFinancialEnvironmentEvaluationable.softmax(action, axis=1)
            elif np_action.ndim==1:
                np_action = BasicWrapperFinancialEnvironmentEvaluationable.softmax(action, axis=1)

        list_flattened_covariance = []
        for key, value in self.unwrapped.dict_ordered_statistics_cov.items():

            #Adding cash as an investment
            tmp_cash_included_cov_matrix = np.zeros((value.shape[0]+1, value.shape[1]+1))
            tmp_cash_included_cov_matrix[1:, 1:] = value
            list_flattened_covariance.append(tmp_cash_included_cov_matrix.flatten())

        np_flattened_covariance = np.array(list_flattened_covariance)

        return calculate_portfolio_variance_from_action(flattened_covariance_matrix=np_flattened_covariance,
                                                 np_portfolio_weight=np_action)

    @staticmethod
    def decompose_observation_wrapper(np_observation, include_risk_penalty_in_state=False):

        if include_risk_penalty_in_state:
            if np_observation.ndim == 2:
                risk_penalty = np_observation[:,0]
                np_observation = np_observation[:,1:]
            elif np_observation.ndim ==1:
                risk_penalty = np_observation[0]
                np_observation = np_observation[1:]

        include_unobservable_market_state_information_for_evaluation = False
        if not include_unobservable_market_state_information_for_evaluation:
            portfolio_wealth, current_state_portfolio_allocation, sampled_output = \
                FinancialMarketsEnv.static_decompose_environment_observation(np_observation=np_observation)

        if include_risk_penalty_in_state:
            return risk_penalty, portfolio_wealth, current_state_portfolio_allocation, sampled_output
        else:
            return portfolio_wealth, current_state_portfolio_allocation, sampled_output

    @staticmethod
    def decompose_observation_wrapper_dict(np_observation, include_risk_penalty_in_state=False):

        if include_risk_penalty_in_state:
            if np_observation.ndim == 2:
                risk_penalty = np_observation[:, 0]
                np_observation = np_observation[:, 1:]
            elif np_observation.ndim == 1:
                risk_penalty = np_observation[0]
                np_observation = np_observation[1:]

        include_unobservable_market_state_information_for_evaluation = False
        if not include_unobservable_market_state_information_for_evaluation:
            tmp_dict = \
                FinancialMarketsEnv.static_decompose_environment_observation_dict(np_observation=np_observation)

        if include_risk_penalty_in_state:
            tmp_dict["risk_penalty"] = risk_penalty
            return tmp_dict
        else:
            return tmp_dict

    def set_memory_mode(self, activate:bool=False):
        print(f'Evaluation_mode_activated ? : {activate}')
        print(len(self.memory_action))
        print(len(self.memory_done))
        print(len(self.memory_full_state))
        print("-------")

        self.memory_mode = activate

    def pop_memory_entries(self):

        tmp_list_memory_action = self.memory_action
        tmp_list_memory_full_state = self.memory_full_state
        tmp_list_memory_done = self.memory_done

        self.memory_action = []
        self.memory_full_state = []
        self.memory_done = []

        return tmp_list_memory_full_state, tmp_list_memory_action, tmp_list_memory_done

    def set_and_return_starting_state_to_evaluate(self, starting_state_to_evaluate: List) -> np.ndarray:
        """
        The evaluation engine only returns the state as a list, but we require a numpy array
        :param starting_state_to_evaluate:
        :return:
        """
        # FIXME WE ARE CURRENTLY JUST PASSING THE OBSERVED STATE AND NOT THE HIDDEN STATE - HOWEVER THIS FUNCTION IS FOR
        # EVALUATION PURPOSES ONLY AND NOT FOR TRAINING

        np_starting_state_to_evaluate = np.array(starting_state_to_evaluate)

        real_hidden_state, portfolio_wealth, current_state_portfolio_allocation, sampled_output = \
            self.unwrapped.decompose_environment_observation(np_starting_state_to_evaluate)

        # np_starting_state_to_evaluate[:self.amount_hidden_states].astype(int)

        #predicted_hidden_market_state
        # setting the startprob for the next step
        self.unwrapped.model.startprob_ = self.unwrapped.model.transmat_[real_hidden_state]

        # We do not have to set the "current_sampled_output" since the perivous sampled output is irrelevant for the state
        self.unwrapped.current_state = real_hidden_state # FIXME so far here we set the predicted state
        # FIXME as the actual state, which is overly optimistic for the estimates
        self.unwrapped.current_sampled_output = sampled_output
        self.unwrapped.current_state_portfolio_allocation = current_state_portfolio_allocation
        self.unwrapped.current_state_portfolio_wealth = portfolio_wealth

        #self.unwrapped.current_state = self.starting_state_to_evaluate  # We have to work here on the
        # #INNER LAYER using self.unwrapped and not self.env.s -> At least for frozen lake
        # see https://stackoverflow.com/questions/53836136/why-unwrap-an-openai-gym
        return self.observation_wrapper(self.unwrapped.create_observation(
            real_hidden_state=self.unwrapped.current_state,
            current_state_portfolio_wealth=self.unwrapped.current_state_portfolio_wealth,
            current_state_portfolio_allocation=self.unwrapped.current_state_portfolio_allocation,
            sampled_output=self.unwrapped.current_sampled_output))


    def reset(self):

        if self.is_one_period_evaluation:
            self.env.reset()
            if self.starting_state_to_evaluate is None:
                raise ValueError(f'evaluation_state is set to None')

            return self.set_and_return_starting_state_to_evaluate(self.starting_state_to_evaluate)
        else:

            if self.randomness_for_risk_penalty:
                self.risk_penalty_current = random.uniform(self.risk_penalty_lower_bound, self.risk_penalty_upper_bound)

            state = self.env.reset()

            if self.memory_mode:
                self.initial_reset_state = state # we have a lag of -1 here

            return self.observation_wrapper(state)

class BasicWrapperFinancialEnvironmentPenaltyState(BasicWrapperFinancialEnvironmentEvaluationable):

    def __init__(self, env, config: EnvContext, apply_softmax_to_actions=False,
                 allow_simplex_action_space_definition=True):
        super().__init__(env=env, config=config, apply_softmax_to_actions=apply_softmax_to_actions,
                         allow_simplex_action_space_definition=allow_simplex_action_space_definition)

        self.include_risk_penalty_in_state = True if config.get("risk_penalty_factor_upper_bound", None) is not None else False

        self.randomness_for_risk_penalty = True

        if self.include_risk_penalty_in_state:
            # Overwriting the previous observation space with the observation space of the wrapped one hot encoded
            # observation space
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(
                self.observation_space.shape[0]+1,), dtype=np.float)


        self.risk_penalty_lower_bound = config.get("risk_penalty_factor_lower_bound")
        self.risk_penalty_upper_bound = config.get("risk_penalty_factor_upper_bound")

        self.risk_penalty_current = None

    def observation_wrapper(self, state):

        if self.unwrapped.include_unobservable_market_state_information_for_evaluation:
            state = state[1:] # cut out the unobservable hidden state so the agent can not see this information

        if self.include_risk_penalty_in_state:
            state = np.concatenate(([self.risk_penalty_current], state))

        return state

    @staticmethod
    def decompose_observation_wrapper(np_observation, include_risk_penalty_in_state=False):

        if include_risk_penalty_in_state:
            if np_observation.ndim == 2:
                risk_penalty = np_observation[:,0]
                np_observation = np_observation[:,1:]
            elif np_observation.ndim ==1:
                risk_penalty = np_observation[0]
                np_observation = np_observation[1:]

        include_unobservable_market_state_information_for_evaluation = False
        if not include_unobservable_market_state_information_for_evaluation:
            portfolio_wealth, current_state_portfolio_allocation, sampled_output = \
                FinancialMarketsEnv.static_decompose_environment_observation(np_observation=np_observation)

        if include_risk_penalty_in_state:
            return risk_penalty, portfolio_wealth, current_state_portfolio_allocation, sampled_output
        else:
            return portfolio_wealth, current_state_portfolio_allocation, sampled_output

    def reset(self):
        if self.randomness_for_risk_penalty:
            self.risk_penalty_current = random.uniform(self.risk_penalty_lower_bound, self.risk_penalty_upper_bound)

        return super().reset()

    def set_randomness_for_risk_penalty(self, randomness_for_risk_penalty):
        self.randomness_for_risk_penalty = randomness_for_risk_penalty

    def set_risk_penalty_current(self, risk_penalty_current):
        self.risk_penalty_current = risk_penalty_current

class BasicWrapperFinancialEnvironmentTrajectoryRisk(BasicWrapperFinancialEnvironmentEvaluationable):

    def __init__(self, env, config: EnvContext, apply_softmax_to_actions=False,
                 allow_simplex_action_space_definition=True):
        super().__init__(env=env, config=config, apply_softmax_to_actions=apply_softmax_to_actions,
                         allow_simplex_action_space_definition=allow_simplex_action_space_definition)

        self.risk_penalty_factor = config.get("risk_penalty_factor")
        self.list_trajectory_rewards = []

    def step(self, action) -> Tuple[np.ndarray, np.ndarray, bool, dict]:

        if self.apply_softmax_to_actions:
            action=action*10
            self.memory_logit.append(action)
            action = BasicWrapperFinancialEnvironmentEvaluationable.softmax(action)

        if self.is_one_period_evaluation:
            if self.action_to_evaluate is None:
                raise ValueError(f'meta_action_to_evaluate is set to None')
            action = self.action_to_evaluate

        next_state, reward, done, info = self.env.step(action)

        self.list_trajectory_rewards.append(reward)

        if self.is_one_period_evaluation:
            done = True

        if self.memory_mode:

            if self.initial_reset_state is not None: # appending the initial state from the reset()
                self.memory_full_state.append(self.initial_reset_state)
                self.initial_reset_state = None # emptying the queue

            if not done:
                self.memory_full_state.append(next_state)  # we have a lag of -1 here
            self.memory_action.append(action)
            self.memory_done.append(done)

        #the reward logic is not just for evaluation, which we use the memory mode for
        self.list_trajectory_rewards.append(reward)

        if done:
            #We add here a scalingfactor, that could go also in the risk_penalty factor, to make it comparable
            reward = reward - (self.risk_penalty_factor * len(self.list_trajectory_rewards)) * np.sqrt(np.var(self.list_trajectory_rewards))

        return self.observation_wrapper(next_state), reward, done, info

    def reset(self):

        self.list_trajectory_rewards = []

        return super().reset()

class BasicWrapperFinancialEnvironment(gym.Wrapper):

    def __init__(self, env, config: EnvContext):
        super().__init__(env)

        self.env = env
        #self.terminal_time_step = 12
        #self.amount_hidden_states = 2
        # unwrapped accesses the wrapped gym
        self.amount_hidden_states = 2 #self.unwrapped.amount_hidden_states

        #card_discrete_action_space = self.action_space.n

        #print("Finance")
        #print(self.action_space.shape)
        # overwriting the action_space (which was discrete to a "meta view", i.e. we return a probablistic action
        # self.action_space = ray.rllib.models.extra_spaces.Simplex(shape=(card_discrete_action_space, ))
        self.action_space = Simplex(shape=self.action_space.shape)

        if self.unwrapped.include_market_state_information:
            # Overwriting the previous observation space with the observation space of the wrapped one hot encoded
            # observation space
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(
                self.observation_space.shape[0]-1+self.amount_hidden_states,), dtype=np.float)
        #print(self.observation_space)
        #print("----------------")
        #else:
        # Otherwise we just take over the
        #    self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(
        #        self.observation_space.shape[0] - 1 + self.amount_hidden_states,), dtype=np.float)

        self.is_one_period_evaluation = False
        self.starting_state_to_evaluate = None
        self.action_to_evaluate = None

    def step(self, action) -> Tuple[np.ndarray, np.ndarray, bool, dict]:

        if self.is_one_period_evaluation:
            if self.action_to_evaluate is None:
                raise ValueError(f'meta_action_to_evaluate is set to None')
            action = self.action_to_evaluate

        next_state, reward, done, info = self.env.step(action)

        if self.is_one_period_evaluation:
            done = True

        return self.observation_wrapper(next_state), reward, done, info

    def observation_wrapper(self, state):
        if self.unwrapped.include_market_state_information:
            predicted_hidden_state = state[0].astype(int)
            np_one_hot_predicted_hidden_state = np_one_hot(predicted_hidden_state, depth=self.amount_hidden_states)

            return np.concatenate((np_one_hot_predicted_hidden_state, state[1:]))
        else:
            return state

    def decompose_model_observation(self, observation: List) -> Tuple:

        np_observation = np.array(observation)

        one_hot_predicted_hidden_state = np_observation[:self.amount_hidden_states].astype(int)

        dummy_state = 0

        np_environment_observation = np.concatenate(([dummy_state], np_observation[self.amount_hidden_states:]))

        _, portfolio_wealth, current_state_portfolio_allocation, sampled_output = \
            self.env.decompose_environment_observation(np_environment_observation)

        return one_hot_predicted_hidden_state, portfolio_wealth, current_state_portfolio_allocation, sampled_output

    def set_and_return_starting_state_to_evaluate(self, starting_state_to_evaluate: List) -> np.ndarray:
        """
        The evaluation engine only returns the state as a list, but we require a numpy array
        :param starting_state_to_evaluate:
        :return:
        """
        # FIXME WE ARE CURRENTLY JUST PASSING THE OBSERVED STATE AND NOT THE HIDDEN STATE - HOWEVER THIS FUNCTION IS FOR
        # EVALUATION PURPOSES ONLY AND NOT FOR TRAINING

        np_starting_state_to_evaluate = np.array(starting_state_to_evaluate)

        one_hot_predicted_hidden_state, portfolio_wealth, current_state_portfolio_allocation, sampled_output = \
            self.decompose_model_observation(np_starting_state_to_evaluate)

        # np_starting_state_to_evaluate[:self.amount_hidden_states].astype(int)

        hidden_market_state_space = gym.spaces.Discrete(self.amount_hidden_states)
        predicted_hidden_market_state = reverse_one_hot_np(np_one_hot=one_hot_predicted_hidden_state,
                                                           original_space=hidden_market_state_space)

        #predicted_hidden_market_state
        # setting the startprob for the next step
        self.unwrapped.model.startprob_ = self.unwrapped.model.transmat_[predicted_hidden_market_state]

        # We do not have to set the "current_sampled_output" since the perivous sampled output is irrelevant for the state
        self.unwrapped.current_state = predicted_hidden_market_state # FIXME so far here we set the predicted state
        # FIXME as the actual state, which is overly optimistic for the estimates
        self.unwrapped.current_sampled_output = sampled_output
        self.unwrapped.current_state_portfolio_allocation = current_state_portfolio_allocation
        self.unwrapped.current_state_portfolio_wealth = portfolio_wealth

        #self.unwrapped.current_state = self.starting_state_to_evaluate  # We have to work here on the
        # #INNER LAYER using self.unwrapped and not self.env.s -> At least for frozen lake
        # see https://stackoverflow.com/questions/53836136/why-unwrap-an-openai-gym
        return self.observation_wrapper(self.unwrapped.create_observation(
            predicted_hidden_state=self.unwrapped.current_state,
            current_state_portfolio_wealth=self.unwrapped.current_state_portfolio_wealth,
            current_state_portfolio_allocation=self.unwrapped.current_state_portfolio_allocation,
            sampled_output=self.unwrapped.current_sampled_output))

    def set_one_period_evaluation_mode(self, is_one_period_evaluation):
        self.is_one_period_evaluation = is_one_period_evaluation

    def set_starting_state_to_evaluate(self, starting_state_to_evaluate):
        self.starting_state_to_evaluate = starting_state_to_evaluate

    def set_action_to_evaluate(self, action_to_evaluate):
        self.action_to_evaluate = action_to_evaluate

    def get_unwrapped_environment_observation_space(self):
        return self.env.observation_space # this returns the initial observation_space before the wrapping

    def reset(self):
        if self.is_one_period_evaluation:

            self.env.reset()

            if self.starting_state_to_evaluate is None:
                raise ValueError(f'evaluation_state is set to None')
            #            return self.observation_wrapper(state)

            return self.set_and_return_starting_state_to_evaluate(self.starting_state_to_evaluate)
        else:
            state = self.env.reset()
            return self.observation_wrapper(state)


class BasicWrapperFinancialEnvironmentShortSelling(gym.Wrapper):

    def __init__(self, env, config: EnvContext):

        super().__init__(env=env)

        #head_factor_list
        self.head_factor_list = config.get("head_factor_list", [])
        self.dict_head_factor = {f'{idx}_head_factor':value for idx, value in enumerate(self.head_factor_list)}
        self.dict_action_mask = config.get("action_mask_dict", {})
        self.dict_uniform_factor = config.get("uniform_factor_dict", None)

        self.dict_trainable_disjoint_mask = config.get("trainable_disjoint_mask_dict", {})
        self.force_single_simplex = config.get("force_single_simplex", False)
        self.force_box_space = config.get("force_box_space", False) #only used for special case
        self.force_discrete = config.get("force_discrete", False) # only used for special case using discrete distribution
        self.force_single_simplex_scaling_dict = config.get("force_single_simplex_scaling_dict", False)  # only used for special case with short selling
        self.force_dict_obs_space = config.get("force_dict_obs_space", False)


        #self.

        #self.actor_conditional_minkowski_encoding = config.get("conditional_minkowski_encoding", False)
        #self.constraints_conditional_minkowski_encoding = config.get("constraints_conditional_minkowski_encoding", False)

        self.actor_conditional_minkowski_encoding_type = config.get("actor_conditional_minkowski_encoding_type", None)
        self.constraints_conditional_minkowski_encoding_type = config.get("constraints_conditional_minkowski_encoding_type", None)

        self.force_parameter_in_action_space = config.get("force_parameter_in_action_space", False) #this is necessary for the autoregressive case due to the RLlib architecture
        #print(trainable_disjoint_mask_dict)

        #if action_mask_list is not None:
        #    self.dict_action_mask = {f'{idx}_action_mask':value for idx, value in enumerate(action_mask_list)}

        print(self.dict_head_factor)

        self.env = env
        #print(config)

        #self.action_space = Simplex(shape=self.action_space.shape)

        self.observation_space = BasicWrapperFinancialEnvironmentShortSelling.generate_observation_space(
            config, amount_assets=self.action_space.shape[0],
            amount_wrapped_observation=self.observation_space.shape[0], force_dict_obs_space=self.force_dict_obs_space)


        print("ENV")
        print(self.observation_space)

        """
        observation_spaces_action_masks = {
            f"{value.split('_')[0]}_action_mask": gym.spaces.Box(0.0, 1.0, shape=(self.action_space.shape[0],))
            for value in self.dict_action_mask.keys()
        }

        observation_spaces_trainable_disjoint_masks = {
            f"{value}": gym.spaces.Box(0.0, 1.0, shape=(self.action_space.shape[0],))
            for value in self.dict_trainable_disjoint_mask.keys()
        }

        observation_spaces_head_factors = {
            f"{value.split('_')[0]}_head_factor": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,))
            for value in self.dict_head_factor.keys()
        }

        observation_spaces = {
            #"0_action_mask": gym.spaces.Box(0.0, 1.0, shape=(self.action_space.shape[0],)),
            #"0_head_factor": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,)), #would be cleaner to do low=0 high=np.inf for the + and low=-np.inf high=0 for the - constraints
            #"1_action_mask": gym.spaces.Box(0.0, 1.0, shape=(self.action_space.shape[0],)),
            #"1_head_factor": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
            #"2_action_mask": gym.spaces.Box(0.0, 1.0, shape=(self.action_space.shape[0],)),
            #"2_head_factor": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
            #"action_mask": gym.spaces.Box(0.0, 1.0, shape=(self.action_space.shape[0],)), # in case we want to predetermine which assets to short and to long
            #"short_head_factor": gym.spaces.Box(low=0, high=np.inf, shape=(1,)), # in case we want to restrict the short selling amount
            #"long_head_factor": gym.spaces.Box(low=0, high=np.inf, shape=(1,)), # in case we want to restrict the long selling amount
            "observations": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(
                    self.observation_space.shape[0],), dtype=np.float),
        }

        #merging all observation spaces together
        observation_spaces = {**observation_spaces, **observation_spaces_action_masks}
        observation_spaces = {**observation_spaces, **observation_spaces_trainable_disjoint_masks}
        observation_spaces = {**observation_spaces, **observation_spaces_head_factors}


        print("---OBS SPACES---")
        print(observation_spaces)
        #self.short_head_factor = config.get("short_head_factor", 0.0)
        #self.long_head_factor = config.get("long_head_factor", 1.0)

        #self.dict_head_factor = {
        #    "0_head_factor" : 1.3,
        #    "1_head_factor" : -0.3,
        #    "2_head_factor" : 0.2,
        #}

        test_mask = np.ones(self.action_space.shape[0])
        test_mask[0] = 0.0
        self.action_mask = test_mask #np.ones(self.action_space.shape[0])

        if len(observation_spaces.keys())>1:
            self.observation_space = gym.spaces.Dict(observation_spaces)
        else:
            self.observation_space = observation_spaces.get("observations") #create simple observation
        """ or None
        #action_spaces = {
        #    '0_allocation': Simplex(shape=self.action_space.shape),  # gym.spaces.Box(low=0, high=1000, shape=(1,)),
        #    '1_allocation': Simplex(shape=self.action_space.shape),
        #    # gym.spaces.Discrete(self.amount_pairs_to_include)
        #}
        self.generate_action_space()


    def generate_action_space(self):

        if not self.force_single_simplex and not self.force_box_space and \
                not self.force_single_simplex_scaling_dict and len(self.dict_head_factor.keys())>0:
            if self.force_discrete:
                # TODO not yet tested
                action_spaces = {  # head_factor amount equals the amount of action spaces
                    f"{value.split('_')[0]}_allocation": gym.spaces.Discrete(self.action_space.shape)
                    for value in self.dict_head_factor.keys()
                }
            else:
                action_spaces = { #head_factor amount equals the amount of action spaces
                    f"{value.split('_')[0]}_allocation": Simplex(shape=self.action_space.shape)
                    for value in self.dict_head_factor.keys()
                }
                if self.dict_uniform_factor is not None:
                    action_spaces_uniform = {
                        f"{key.split('_')[0]}_uniform_factor": gym.spaces.Box(low=0, high=value, shape=(1,))
                        for key, value in self.dict_uniform_factor.items() if value is not None}
                    action_spaces = {**action_spaces, **action_spaces_uniform}
                if self.force_parameter_in_action_space:
                    action_spaces_parameters = {  # head_factor amount equals the amount of action spaces
                        f"{value.split('_')[0]}_parameter": gym.spaces.Box(low=0, high=np.inf,  shape=self.action_space.shape, dtype=np.float)
                        for value in self.dict_head_factor.keys()
                    }
                    action_spaces = {**action_spaces, **action_spaces_parameters}

            self.action_space = gym.spaces.Dict(action_spaces)
        elif self.force_single_simplex_scaling_dict and len(self.dict_head_factor.keys())>0:

            scaling_factor_value = 1.3
            action_spaces = {
                'action_encoding': Simplex(shape=self.action_space.shape),
                'scaling_factor_0' : gym.spaces.Box(low=scaling_factor_value, high=scaling_factor_value, shape=(1,), dtype=np.float)
            }
            self.action_space = gym.spaces.Dict(action_spaces)

            #self.action_space = Simplex(shape=self.action_space.shape)
        elif self.force_box_space and len(self.dict_head_factor.keys())>0:
            #forces box space with upper and lower bound
            print("HIT BOX SPACE")
            low_val = -0.3
            high_val = 1.3
            self.action_space = gym.spaces.Box(low=low_val, high=high_val, shape=self.action_space.shape, dtype=np.float)
        elif self.force_single_simplex:
            self.action_space = Simplex(shape=self.action_space.shape)
        else:
            self.action_space = Simplex(shape=self.action_space.shape)

    @staticmethod
    def generate_observation_space(config, amount_assets, amount_wrapped_observation, force_dict_obs_space=False):

        head_factor_list = config.get("head_factor_list", [])
        tmp_dict_head_factor = {f'{idx}_head_factor': value for idx, value in enumerate(head_factor_list)}
        tmp_dict_action_mask = config.get("action_mask_dict", {})
        tmp_dict_trainable_disjoint_mask = config.get("trainable_disjoint_mask_dict", {})


        observation_spaces_action_masks = {
            f"{value.split('_')[0]}_action_mask": gym.spaces.Box(0.0, 1.0, shape=(amount_assets,))
            for value in tmp_dict_action_mask.keys()
        }

        observation_spaces_trainable_disjoint_masks = {
            f"{value}": gym.spaces.Box(0.0, 1.0, shape=(amount_assets,))
            for value in tmp_dict_trainable_disjoint_mask.keys()
        }

        observation_spaces_head_factors = {
            f"{value.split('_')[0]}_head_factor": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,))
            for value in tmp_dict_head_factor.keys()
        }

        observation_spaces = {
            "observations": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(
                    amount_wrapped_observation,), dtype=np.float),
        }

        observation_spaces = {**observation_spaces, **observation_spaces_action_masks}
        observation_spaces = {**observation_spaces, **observation_spaces_trainable_disjoint_masks}
        observation_spaces = {**observation_spaces, **observation_spaces_head_factors}

        if len(observation_spaces.keys())>1 or force_dict_obs_space:
            gym_observation_spaces = gym.spaces.Dict(observation_spaces)
        else:
            gym_observation_spaces = observation_spaces.get("observations") #create simple observation

        return gym_observation_spaces

    @staticmethod
    def decompose_observation_wrapper_dict(np_observation_full, config):

        include_risk_penalty_in_state = config.get("include_risk_penalty_in_state", False)

        if include_risk_penalty_in_state:
            if np_observation_full.ndim == 2:
                risk_penalty = np_observation_full[:, 0]
                np_observation = np_observation_full[:, 1:]
            elif np_observation_full.ndim == 1:
                risk_penalty = np_observation_full[0]
                np_observation = np_observation_full[1:]

        # reverse engineer amount_assets
        head_factor_list = config.get("head_factor_list", [])
        tmp_dict_head_factor = {f'{idx}_head_factor': value for idx, value in enumerate(head_factor_list)}
        tmp_dict_action_mask = config.get("action_mask_dict", {})
        tmp_dict_trainable_disjoint_mask = config.get("trainable_disjoint_mask_dict", {})
        force_dict_obs_space = config.get("force_dict_obs_space", False)

        amount_single_entries = 0
        amount_asset_input_size_entries = 0

        # due to wrapped environment
        amount_single_entries +=1 #due to obs
        amount_asset_input_size_entries += 2 #due to obs

        # constraint related
        amount_single_entries += len(tmp_dict_head_factor.keys())
        amount_asset_input_size_entries += len(tmp_dict_action_mask.keys())
        amount_asset_input_size_entries += len(tmp_dict_trainable_disjoint_mask.keys())

        if np_observation_full.ndim == 2:
            amount_assets = int((np_observation_full.shape[1] - amount_single_entries)/amount_asset_input_size_entries)
            tmp_gym_obs_space = BasicWrapperFinancialEnvironmentShortSelling.generate_observation_space(config, amount_assets=amount_assets,
                                                                                    amount_wrapped_observation=(2*amount_assets+1),
                                                                                    force_dict_obs_space=force_dict_obs_space)
            if not isinstance(tmp_gym_obs_space, gym.spaces.Dict):
                return np_observation_full
            else:
                obs_dict = {}
                current_idx_counter = 0
                # print(self.observation_space.contains("observations"))
                for key_val in tmp_gym_obs_space.spaces.keys():
                    substring_action_mask = "action_mask"
                    substring_trainable_disjoint_mask = "trainable_disjoint_mask"
                    substring_head_factor = "head_factor"
                    if key_val == "observations":
                        tmp_dict = \
                            FinancialMarketsEnv.static_decompose_environment_observation_dict(
                                np_observation=np_observation_full[:,current_idx_counter:current_idx_counter+(2*amount_assets+1)])
                        current_idx_counter +=(2*amount_assets+1)
                        #obs_dict["observations"] = raw_observation
                        obs_dict = {**obs_dict, **tmp_dict}
                    elif re.search(substring_action_mask, key_val):
                        obs_dict[key_val] = np_observation_full[:,current_idx_counter:current_idx_counter+amount_assets]
                        current_idx_counter += amount_assets
                    elif re.search(substring_head_factor, key_val):
                        obs_dict[key_val] = np_observation_full[:,current_idx_counter:current_idx_counter+1]
                        current_idx_counter += 1
                    elif re.search(substring_trainable_disjoint_mask, key_val):
                        obs_dict[key_val] = np_observation_full[:,
                                            current_idx_counter:current_idx_counter + amount_assets]
                        current_idx_counter += amount_assets
                return obs_dict
        elif np_observation_full.ndim == 1:
            raise NotImplementedError


    def process_action(self, action):

        list_allocation = []
        if isinstance(action, Dict):
            if self.force_single_simplex_scaling_dict:
                fixed_scaling_factor = 1.3  # None
                if fixed_scaling_factor is None:
                    scaling_factor = abs(action.get("scaling_factor_0")) + abs(-(action.get("scaling_factor_0")-1))
                else:
                    scaling_factor = abs(fixed_scaling_factor) + abs(-(fixed_scaling_factor - 1))
                np_long = np.array(self.dict_action_mask.get("0_action_mask")) * scaling_factor
                np_short = - np.array(self.dict_action_mask.get("1_action_mask")) * scaling_factor

                np_long_leg = np_long * action.get("action_encoding")
                np_short_leg = np_short * action.get("action_encoding")

                list_allocation.append(np_long_leg)
                list_allocation.append(np_short_leg)
            else:
                if self.actor_conditional_minkowski_encoding_type is not None:
                    #special case
                    tmp_dict_actions = {}
                    for key, value in action.items():
                        tmp_dict_actions[key]=value
                    if self.constraints_conditional_minkowski_encoding_type is None:  ##Old standard way
                        merged_action = convert_conditional_minkowski_encoding_to_action(dict_raw_actions=tmp_dict_actions,
                                                                            dict_action_mask=self.dict_action_mask,
                                                                            head_factor_list=self.head_factor_list)
                    else:
                        merged_action = convert_conditional_minkowski_encoding_to_action(
                            dict_raw_actions=tmp_dict_actions,
                            dict_action_mask=self.dict_action_mask,
                            head_factor_list=self.head_factor_list,
                            dict_uniform_factor=self.dict_uniform_factor,
                            encoding_type=self.actor_conditional_minkowski_encoding_type)

                    return merged_action
                else: #standard case
                    for key, value in action.items():
                        str_index_allocation = key.split('_')[0]
                        tmp_action = value * self.dict_head_factor.get(f"{str_index_allocation}_head_factor")
                        list_allocation.append(tmp_action)
        else:
            if False: #self.force_single_simplex_scaling_dict: #specical case
                # scale and adapt the sign
                #print(self.dict_action_mask)
                scaling_factor = abs(self.dict_head_factor.get(f"0_head_factor"))+abs(self.dict_head_factor.get(f"1_head_factor"))
                np_long = np.array(self.dict_action_mask.get("0_action_mask"))*scaling_factor
                np_short = - np.array(self.dict_action_mask.get("1_action_mask"))*scaling_factor

                np_long_leg = np_long * action
                np_short_leg = np_short * action

                list_allocation.append(np_long_leg)
                list_allocation.append(np_short_leg)
            elif self.force_single_simplex:
                return action
            else:
                list_allocation.append(action)

        allocation_matrix = np.stack(list_allocation, axis=0)

        action_merged = np.sum(allocation_matrix, axis=0)
        return action_merged

    def step(self, action) -> Tuple[np.ndarray, np.ndarray, bool, dict]:

        action_merged = self.process_action(action)

        next_state, reward, done, info = self.env.step(action_merged)
        return self.create_wrapped_observation(next_state), reward, done, info

    def reset(self) -> np.ndarray:
        state = self.env.reset()
        return self.create_wrapped_observation(state)

    def create_wrapped_observation(self, raw_observation):
        #state = raw_observation
        if not isinstance(self.observation_space, gym.spaces.Dict):
            return raw_observation
        else:
            obs_dict = {}
            #print(self.observation_space.contains("observations"))
            for key_val in self.observation_space.spaces.keys():
                #fullstring = "0_action_mask"
                substring_action_mask = "action_mask"
                substring_trainable_disjoint_mask = "trainable_disjoint_mask"
                substring_head_factor = "head_factor"

                if key_val=="observations":
                    obs_dict["observations"] = raw_observation
                elif re.search(substring_action_mask, key_val):
                    if key_val in self.dict_action_mask:
                        obs_dict[key_val] = np.array(self.dict_action_mask.get(key_val))
                    else:
                        raise ValueError(f"Unknown action mask string {key_val} - TODO implement also some missing action masks to be a valid input")
                    #obs_dict[key_val] = self.action_mask  # ensure that all outputs are np.arrays
                elif re.search(substring_head_factor, key_val):
                    if key_val in self.dict_head_factor:
                        obs_dict[key_val] = np.array([self.dict_head_factor.get(key_val)])
                    else:
                        raise ValueError(f"Unknown head factor string {key_val}")
                elif re.search(substring_trainable_disjoint_mask, key_val):
                    if key_val in self.dict_trainable_disjoint_mask:
                        obs_dict[key_val] = np.array(self.dict_trainable_disjoint_mask.get(key_val))
                    else:
                        raise ValueError(f"Unknown dict trainable string {key_val}")

            return obs_dict
