import numpy as np

from hmmlearn import hmm
import json
import gym
from typing import Tuple, Dict
import importlib.resources
import json

class FinancialMarketsEnv(gym.Env):

    def __init__(self, terminal_time_step=12, include_cash_asset: bool=True,
                 include_unobservable_market_state_information_for_evaluation: bool=True,
                 data_set_name: str="model_parameter_data_set_G_markov_states_2_2",#"model_parameter_data_set_A_currency_USD_markov_states_2",
                 one_sided_spread_fee: float= 0.000,
                 short_selling_yearly_fee: float= 0.03,
                 **kwargs):

        self.current_hidden_market_state = None
        self.current_predicted_market_state = None
        self.current_sampled_output = None
        self.current_startprob = None


        self.current_time_step = 0
        self.terminal_time_step = terminal_time_step

        self.model_choice = data_set_name

        self.amount_states, amount_features, means_, covars_, transmat_, self.initial_startprob_ = \
            self.initializing_hmm_model_parameters()

        self.dict_ordered_returns, self.dict_ordered_statistics_mean, self.dict_ordered_statistics_cov = \
            self.initializing_backtesting_parameters()

        self.model = hmm.GaussianHMM(n_components=self.amount_states, covariance_type="full")

        self.model.startprob_ = self.initial_startprob_
        self.model.transmat_ = transmat_
        self.model.means_ = means_
        self.model.covars_ = covars_

        self.include_cash_asset = include_cash_asset
        self.include_unobservable_market_state_information_for_evaluation = include_unobservable_market_state_information_for_evaluation

        self.amount_assets = amount_features

        if self.include_cash_asset:
            self.amount_assets+=1

        self.is_backtesting_mode = False

        self.ask_spread_vector = np.ones((self.amount_assets), dtype=float)*one_sided_spread_fee  #this is for how much we can buy an asset
        self.ask_spread_vector[0] = 0.0 # buying cash does not cost anything (but we have to pay the bid spread of the asset we go out of)

        self.bid_spread_vector = np.ones((self.amount_assets),
                                         dtype=float) * one_sided_spread_fee  # this is for how much we can buy an asset
        self.bid_spread_vector[
            0] = 0.0  # selling cash does not cost anything (but we have to pay the ask spread of the asset we go into in)

        self.short_selling_yearly_fee_vector = np.ones((self.amount_assets), dtype=float)*short_selling_yearly_fee

        # portfolio specific state information
        self.current_state_portfolio_allocation = np.zeros((self.amount_assets), dtype=float)  # numpy vector
        self.current_state_portfolio_allocation[0] = 1.0 # setting initial allocation to 100% cash
        self.current_state_portfolio_wealth = 100  # None # float

        # Portfolio allocation which has to sum up to 1, i.e. action_space is a simplex
        #self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.amount_assets,), dtype=np.float)

        #Allowing for short selling
        self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.amount_assets,), dtype=np.float)

        # Box is for R valued observation space while MultiDiscrete is for multidimensional discrete action space
        amount_observations = self.amount_assets + self.amount_assets + 1
        # +1 for portfolio wealth
        # +self.amount_assets for currently held portfolio allocation
        if include_unobservable_market_state_information_for_evaluation:
            amount_observations += 1
        # +1 for the hidden state //give back portfolio state and portfolio wealth

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(amount_observations,), dtype=np.float)


    def set_backtesting_mode(self, is_backtesting_mode):
        self.is_backtesting_mode = is_backtesting_mode

    def initializing_hmm_model_parameters(self)-> Tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:

        #https://stackoverflow.com/questions/779495/access-data-in-package-subdirectory
        with importlib.resources.open_text("financial_markets_gym.envs.models", f"{self.model_choice}.json") as file:
            parameter_dict = json.load(file)

        means_ = np.array(parameter_dict.get("model.means_"))

        amount_states = means_.shape[0]
        amount_features = means_.shape[1]

        transmat_ = np.array(parameter_dict.get("model.transmat_"))
        covars_ = np.array(parameter_dict.get("model.covars_"))
        startprob_ = np.array(parameter_dict.get("model.startprob_"))

        return amount_states, amount_features, means_, covars_, transmat_, startprob_

    def initializing_backtesting_parameters(self):# -> Dict[str: np.ndarray]:

        try:
            backtesting_file = f"{self.model_choice.replace('parameter', 'backtesting')}.json"
            with importlib.resources.open_text("financial_markets_gym.envs.models", f"{backtesting_file}") as file:
                backtesting_dict = json.load(file)

            tmp_dict_ordered_returns = {}
            for key, value in backtesting_dict.get("data_ordered_returns").items():
                tmp_dict_ordered_returns[key] =  np.array(value)

            tmp_dict_ordered_statistics_mean = {}
            for key, value in backtesting_dict.get("data_statistics_mean").items():
                tmp_dict_ordered_statistics_mean[key] = np.array(value)

            tmp_dict_ordered_statistics_cov = {}
            for key, value in backtesting_dict.get("data_statistics_cov").items():
                tmp_dict_ordered_statistics_cov[key] = np.array(value)

        except FileNotFoundError as e:
            backtesting_file = f"{self.model_choice.replace('parameter', 'backtesting')}.json"

            print(f"WARNING: No backtesting file {backtesting_file} was found.")
            tmp_dict_ordered_returns = None
            tmp_dict_ordered_statistics_mean = None
            tmp_dict_ordered_statistics_cov = None

        return tmp_dict_ordered_returns, tmp_dict_ordered_statistics_mean, tmp_dict_ordered_statistics_cov

    def render(self):
        pass

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:

        # calculate trading fees:
        trading_delta_vector = action - self.current_state_portfolio_allocation # increases will be positive, decreases negative
        trading_delta_vector_buys = np.maximum(trading_delta_vector, np.zeros_like(trading_delta_vector))
        trading_delta_vector_sells = -np.minimum(trading_delta_vector, np.zeros_like(trading_delta_vector)) # we define it positive

        trading_fee = np.dot(trading_delta_vector_buys, self.ask_spread_vector) + \
                      np.dot(trading_delta_vector_sells, self.bid_spread_vector)

        # short selling fees:
        short_positions = -np.minimum(action, np.zeros_like(action)) # we define the short position positive
        short_fee = np.dot(short_positions, self.short_selling_yearly_fee_vector/12.0)

        if self.is_backtesting_mode:
            sampled_output = self.dict_ordered_returns.get(str(self.current_time_step))
            sampled_output = np.expand_dims(sampled_output, axis=0)
            hidden_state = np.array([0]) #dummy value
        else:
            sampled_output, hidden_state = self.model.sample(1)

        self.current_predicted_market_state = self.model.predict(sampled_output).item()


        if self.include_cash_asset:
            cash_return = [0.0]
            sampled_output = np.concatenate(([cash_return], sampled_output), axis=1)

        self.current_sampled_output = sampled_output.flatten()
        self.current_hidden_market_state = hidden_state.item()


        reward = np.dot(action, self.current_sampled_output) - trading_fee - short_fee

        market_movement_adjustment = np.ones_like(self.current_sampled_output)+self.current_sampled_output

        self.current_state_portfolio_allocation = (action*market_movement_adjustment)/np.sum(action*market_movement_adjustment)
        self.current_state_portfolio_wealth *= (1+reward)

        # setting the startprob for the next step
        self.model.startprob_ = self.model.transmat_[self.current_hidden_market_state]

        observation = self.create_observation(real_hidden_state=self.current_hidden_market_state,
                                              current_state_portfolio_wealth=self.current_state_portfolio_wealth,
                                              current_state_portfolio_allocation=self.current_state_portfolio_allocation,
                                              sampled_output=self.current_sampled_output)

        if self.current_time_step >= self.terminal_time_step:
            done = True
        else:
            done = False


        self.current_time_step += 1

        return observation, reward, done, {}


    def create_observation(self, real_hidden_state: int, current_state_portfolio_wealth:float,
                                              current_state_portfolio_allocation: np.ndarray,
                                              sampled_output: np.ndarray) -> np.ndarray:
        if self.include_unobservable_market_state_information_for_evaluation:
            return np.concatenate(([real_hidden_state], [current_state_portfolio_wealth],
                                   current_state_portfolio_allocation, sampled_output))
        else:
            return np.concatenate(([current_state_portfolio_wealth],
                                   current_state_portfolio_allocation, sampled_output))

    def decompose_environment_observation(self, np_observation: np.ndarray) -> Tuple:

        if self.include_unobservable_market_state_information_for_evaluation:
            real_hidden_state = int(np_observation[0])
            portfolio_wealth = np_observation[1]
            current_state_portfolio_allocation = np_observation[2:2 + self.amount_assets]
            sampled_output = np_observation[2 + self.amount_assets:]

            return real_hidden_state, portfolio_wealth, current_state_portfolio_allocation, sampled_output
        else:
            portfolio_wealth = np_observation[0]
            current_state_portfolio_allocation = np_observation[1:1 + self.amount_assets]
            sampled_output = np_observation[1 + self.amount_assets:]

            return portfolio_wealth, current_state_portfolio_allocation, sampled_output

    @staticmethod
    def check_all_elements_included(np_observation, last_read_element_index):
        if np_observation.ndim == 2:
            assert np_observation.shape[1] == last_read_element_index
        elif np_observation.ndim == 1:
            assert np_observation.shape[0] == last_read_element_index

    @staticmethod
    def static_decompose_environment_observation_dict(np_observation: np.ndarray,
                                                      include_unobservable_market_state_information_for_evaluation: bool = False) -> Tuple:
        if np_observation.ndim == 2:
            # Batch case
            amount_assets = FinancialMarketsEnv.calculate_amount_assets(np_observation[0, :].flatten(),
                                                                        include_unobservable_market_state_information_for_evaluation)
            if include_unobservable_market_state_information_for_evaluation:
                real_hidden_state = int(np_observation[:, 0])
                portfolio_wealth = np_observation[:, 1]
                current_state_portfolio_allocation = np_observation[:, 2:2 + amount_assets]
                final_index_element = 2 + amount_assets + amount_assets
                prev_observed_returns = np_observation[:, 2 + amount_assets:final_index_element]
                FinancialMarketsEnv.check_all_elements_included(np_observation, final_index_element)
                tmp_dict = {"real_hidden_state": real_hidden_state,
                            "portfolio_wealth": portfolio_wealth,
                            "current_state_portfolio_allocation": current_state_portfolio_allocation,
                            "prev_observed_returns": prev_observed_returns}
                # return real_hidden_state, portfolio_wealth, current_state_portfolio_allocation, prev_observed_returns
                return tmp_dict
            else:
                portfolio_wealth = np_observation[:, 0]
                current_state_portfolio_allocation = np_observation[:, 1:1 + amount_assets]
                final_index_element = 1 + amount_assets+amount_assets
                prev_observed_returns = np_observation[:, 1 + amount_assets:final_index_element]
                FinancialMarketsEnv.check_all_elements_included(np_observation, final_index_element)
                tmp_dict = {"portfolio_wealth": portfolio_wealth,
                            "current_state_portfolio_allocation": current_state_portfolio_allocation,
                            "prev_observed_returns": prev_observed_returns}
                # return portfolio_wealth, current_state_portfolio_allocation, sampled_output
                return tmp_dict
        elif np_observation.ndim == 1:
            amount_assets = FinancialMarketsEnv.calculate_amount_assets(np_observation,
                                                                        include_unobservable_market_state_information_for_evaluation)

            if include_unobservable_market_state_information_for_evaluation:
                real_hidden_state = int(np_observation[0])
                portfolio_wealth = np_observation[1]
                current_state_portfolio_allocation = np_observation[2:2 + amount_assets]
                final_index_element = 2 + amount_assets + amount_assets
                prev_observed_returns = np_observation[2 + amount_assets:final_index_element]
                FinancialMarketsEnv.check_all_elements_included(np_observation, final_index_element)

                tmp_dict = {"real_hidden_state": real_hidden_state,
                            "portfolio_wealth": portfolio_wealth,
                            "current_state_portfolio_allocation": current_state_portfolio_allocation,
                            "prev_observed_returns": prev_observed_returns}
                # return real_hidden_state, portfolio_wealth, current_state_portfolio_allocation, prev_observed_returns
                return tmp_dict
            else:
                portfolio_wealth = np_observation[0]
                current_state_portfolio_allocation = np_observation[1:1 + amount_assets]
                final_index_element = 1 + amount_assets + amount_assets
                prev_observed_returns = np_observation[1 + amount_assets:final_index_element]
                FinancialMarketsEnv.check_all_elements_included(np_observation, final_index_element)

                tmp_dict = {"portfolio_wealth": portfolio_wealth,
                            "current_state_portfolio_allocation": current_state_portfolio_allocation,
                            "prev_observed_returns": prev_observed_returns}
                # return portfolio_wealth, current_state_portfolio_allocation, sampled_output
                return tmp_dict

    @staticmethod
    def static_decompose_environment_observation(np_observation: np.ndarray,
                                                 include_unobservable_market_state_information_for_evaluation: bool = False) -> Tuple:
        amount_assets = FinancialMarketsEnv.calculate_amount_assets(np_observation,
                                                                    include_unobservable_market_state_information_for_evaluation)

        if include_unobservable_market_state_information_for_evaluation:
            real_hidden_state = int(np_observation[0])
            portfolio_wealth = np_observation[1]
            current_state_portfolio_allocation = np_observation[2:2 + amount_assets]
            sampled_output = np_observation[2 + amount_assets:]

            return real_hidden_state, portfolio_wealth, current_state_portfolio_allocation, sampled_output
        else:
            portfolio_wealth = np_observation[0]
            current_state_portfolio_allocation = np_observation[1:1 + amount_assets]
            sampled_output = np_observation[1 + amount_assets:]

            return portfolio_wealth, current_state_portfolio_allocation, sampled_output

    @staticmethod
    def calculate_amount_assets(np_observation: np.ndarray,
                                include_unobservable_market_state_information_for_evaluation: bool = False) -> int:
        additional_fields = 0
        # portfolio_wealth
        additional_fields += 1
        if include_unobservable_market_state_information_for_evaluation:
            additional_fields += 1

        # since we have two full amount_asset measures

        return int((np_observation.shape[0] - additional_fields) / 2)

    def reset(self):

        self.current_state_portfolio_allocation = np.zeros((self.amount_assets), dtype=float)  # numpy vector
        self.current_state_portfolio_allocation[0] = 1.0 # setting initial allocation to 100% cash
        self.current_state_portfolio_wealth = 100  # None # float

        self.current_hidden_market_state = None
        self.model.startprob_ = self.initial_startprob_

        sampled_output, hidden_state = self.model.sample(1)

        self.current_predicted_market_state = self.model.predict(sampled_output).item()

        if self.include_cash_asset:
            cash_return = [0.0]
            sampled_output = np.concatenate(([cash_return], sampled_output), axis=1)

        self.current_sampled_output = sampled_output.flatten()
        self.current_hidden_market_state = hidden_state.item()

        # setting the startprob for the next step
        self.model.startprob_ = self.model.transmat_[self.current_hidden_market_state]

        # setting initial time step
        self.current_time_step = 0

        return self.create_observation(real_hidden_state=self.current_hidden_market_state,
                                              current_state_portfolio_wealth=self.current_state_portfolio_wealth,
                                              current_state_portfolio_allocation=self.current_state_portfolio_allocation,
                                              sampled_output=self.current_sampled_output)
