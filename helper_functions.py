import torch
from gym.spaces import Discrete, MultiDiscrete
from ray.rllib.utils.framework import try_import_torch
import numpy as np
from typing import Tuple
import pandas as pd

torch, nn = try_import_torch()
from ray.rllib.policy.sample_batch import SampleBatch
from custom_keys import Postprocessing_Custom
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.pyplot as plt
import os
import gym
from ray.rllib.utils.spaces.simplex import Simplex

from distribution_autoregressive_custom_types import TorchAutoregressiveDirichletDistributionS4_U1, TorchAutoregressiveDirichletDistributionS4

def generate_seq_length(train_batch) -> torch.Tensor:
    if isinstance(train_batch[SampleBatch.REWARDS], torch.Tensor):
        is_input_tensor = True
        if SampleBatch.SEQ_LENS in train_batch:
            return train_batch[SampleBatch.SEQ_LENS]
        else:
            return torch.masked_select(train_batch[Postprocessing_Custom.SEQ_LENS_DUMMY].type(torch.int32),
                                       train_batch[SampleBatch.DONES].type(torch.bool))
    else:
        is_input_tensor = False
        if SampleBatch.SEQ_LENS in train_batch:
            return train_batch[SampleBatch.SEQ_LENS] #numpy
        else:
            if np.count_nonzero(train_batch[SampleBatch.REWARDS]) == 0: # To cover the initializing phase
                tmp_mask = np.zeros_like(train_batch[SampleBatch.REWARDS])
                tmp_mask[-1] = 1
                return train_batch[Postprocessing_Custom.SEQ_LENS_DUMMY].astype(int)[
                                       tmp_mask.astype(bool)]
            return train_batch[Postprocessing_Custom.SEQ_LENS_DUMMY].astype(int)[
                                       train_batch[SampleBatch.DONES].astype(bool)]


class FirstSecondMomentDataset(torch.utils.data.Dataset):
    def __init__(self, window_size: int = 12, prediction_length: int = 1, auto_regressive: bool = True,
                 np_first_second_moment_information: np.ndarray=None, seq_lens: np.ndarray=None,
                 moment_model: torch.nn.Module=None):


        self.list_samples = convert_first_moment_and_second_moment_information_batch_to_samples(
            np_first_second_moment_information = np_first_second_moment_information,
            seq_lens=seq_lens, moment_model=moment_model, window_size=window_size)

        #step_size = 4000
        #self.x = torch.linspace(start=0, end=10 * 2 * torch.pi, steps=step_size).unsqueeze(dim=-1)
        #tmp_list = []
        #tmp_list.append(torch.normal(2.0, 1.0, size=(step_size, 1)))
        #tmp_list.append(torch.normal(8.0, 1.0, size=(step_size, 1)))
        #self.y = torch.concat(tmp_list, dim=1)
        # print(self.y.shape)
        # self.y = self.x.sin() + noise_level * torch.rand_like(self.x)
        self.window_size = window_size
        self.horizon = prediction_length
        self.auto_regressive = auto_regressive

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor]:

        start = 0
        input_end = self.window_size
        end = start + self.window_size + self.horizon
        sample = self.list_samples[item]
        #inp = self.y if self.auto_regressive else self.x

        # print(inp[start:input_end].shape)
        # print("GET")
        # print(inp[start:input_end])
        # print(self.y[input_end:end])
        # print("END")
        # The decoder input target needs to be shifted by 1 in comparison to the decoder output target
        # see https://arxiv.org/abs/2001.08317
        # return inp[start:input_end], self.y[(input_end-1):end]
        return sample[start:input_end], sample[(input_end - 1):end]

    def __len__(self):
        #return len(self.x) - self.window_size - self.horizon  # amount of starting idx you are allowed to pick
        return len(self.list_samples)

def add_pre_padding(np_array: np.ndarray, initial_padding: int):
    tmp_zeros = np.zeros((initial_padding, np_array.shape[1]))
    return np.concatenate((tmp_zeros, np_array), axis=0)

def add_post_padding(np_array: np.ndarray):
    # tmp_zeros = np.zeros((initial_padding, np_array.shape[1]))
    tmp_dummy = np.expand_dims(np_array[-1, :], 0)  # copy last row

    return np.concatenate((np_array, tmp_dummy), axis=0)


def generate_moving_window_samples(np_array: np.ndarray, moment_submodel: torch.nn.Module,
                                   time_steps_per_sample_incl_target: int = (12 + 1)):
    sample_list = []
    np_idx_list = np.arange(np_array.shape[0])
    sliding_view = sliding_window_view(np_idx_list, time_steps_per_sample_incl_target)
    for idx_sliding in sliding_view:
        sample_list.append(torch.from_numpy(np_array[idx_sliding]).float().to(
            moment_submodel.availabe_device))
    return sample_list


def convert_first_moment_and_second_moment_information_batch_to_samples(np_first_second_moment_information: np.ndarray,
                                                                        seq_lens: np.ndarray, moment_model: torch.nn.Module, window_size: int):
    np_seq_lens = seq_lens #seq_lens.cpu().detach().numpy()
    np_seq_lens_cum_sum = np.cumsum(np_seq_lens)
    np_split_trajectory_information = np.vsplit(np_first_second_moment_information, np_seq_lens_cum_sum[
                                                           :-1])  # We do not need the last element, otherwise it will split one more time a null array
    list_np_split_trajectory_information = [add_pre_padding(entry, initial_padding=(window_size-1)) for
                                            entry in np_split_trajectory_information]

    list_np_split_trajectory_information = [add_post_padding(entry) for
                                            entry in list_np_split_trajectory_information]

    list_total_samples = []
    for np_split_trajectory_information in list_np_split_trajectory_information:
        list_total_samples.extend(generate_moving_window_samples(np_split_trajectory_information, moment_submodel=moment_model))

    return list_total_samples


def calculate_sharpe_ratio(np_total_period_excess_return, np_total_period_excess_return_variance):

    assert np_total_period_excess_return.size == 1
    assert np_total_period_excess_return_variance.size == 1

    #Sharpe ratio is defined as the access return divided by the access returns standard deviation
    return np_total_period_excess_return/np.sqrt(np_total_period_excess_return_variance)


def update_state(model_config, state, prev_model_memory_out=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = model_config
    if state is None:
        initial_state = []
        for i in range(cfg["attention_num_transformer_units"]):
            # The first 1 comes from the trajectory and is only valid in its batched form later we have to batch these
            initial_state.append(torch.zeros(1, cfg["attention_memory_inference"], cfg[
                "attention_dim"], device=device))
        return initial_state
    else:
        state = [torch.cat((state[i], torch.unsqueeze(prev_model_memory_out[i], 0)), 1)[:, 1:, :] for i in
                 range(cfg["attention_num_transformer_units"])]
        return state


def generate_trainable_dataset(moment_submodel, train_batch):

    if isinstance(train_batch[SampleBatch.REWARDS], torch.Tensor):
        is_input_tensor = True
    else:
        is_input_tensor = False

    if moment_submodel.moment_model_output_aggregated_portfolio:
        raise NotImplementedError #TODO
    else:
        if is_input_tensor:
            previously_observed_returns = train_batch[Postprocessing_Custom.REWARDS_DETAILS].cpu().detach().numpy()
            seq_lens = generate_seq_length(train_batch).cpu().detach().numpy()
        else:
            previously_observed_returns = train_batch[Postprocessing_Custom.REWARDS_DETAILS]
            seq_lens = generate_seq_length(train_batch)#train_batch.get(SampleBatch.SEQ_LENS) #generate_seq_length(train_batch)

        np_output = construct_first_moment_and_second_moment_information(np_observation=previously_observed_returns)

        dataset = FirstSecondMomentDataset(window_size=12, prediction_length=1,
                                           np_first_second_moment_information=np_output, seq_lens=seq_lens,
                                           moment_model=moment_submodel)
        return dataset

def construct_econ_returns(moment_submodel, train_batch):

    if isinstance(train_batch[SampleBatch.REWARDS], torch.Tensor):
        is_input_tensor = True
    else:
        is_input_tensor = False

    if moment_submodel.moment_model_output_aggregated_portfolio:
        if not is_input_tensor:
            torch_econ_first_moment_return_info = torch.from_numpy(train_batch[SampleBatch.REWARDS]).float().to(
                moment_submodel.availabe_device)
        else:
            torch_econ_first_moment_return_info = train_batch[SampleBatch.REWARDS]
        torch_econ_second_moment_return_info = torch.pow(torch_econ_first_moment_return_info, 2.0)
    else:
        if is_input_tensor:
            #np_obs = train_batch["obs"].cpu().detach().numpy()
            previously_observed_returns = train_batch[Postprocessing_Custom.REWARDS_DETAILS].cpu().detach().numpy()
        else:
            #np_obs = train_batch["obs"]
            previously_observed_returns = train_batch[Postprocessing_Custom.REWARDS_DETAILS]

        amount_assets = moment_submodel.number_input_assets#action_space.shape[0]

        #portfolio_wealth, current_state_portfolio_allocation, sampled_output = \
        #    BasicWrapperFinancialEnvironmentPenaltyState.decompose_observation_wrapper(np_obs)

        np_output = construct_first_moment_and_second_moment_information(np_observation=previously_observed_returns)

        torch_econ_first_moment_return_info = torch.from_numpy(np_output[:, :amount_assets]).float().to(
            moment_submodel.availabe_device)

        torch_econ_second_moment_return_info = torch.from_numpy(np_output[:, amount_assets:]).float().to(
            moment_submodel.availabe_device)

    return torch_econ_first_moment_return_info, torch_econ_second_moment_return_info



def calculate_estimated_portfolio_variance_from_action(np_action_flattened_covariance_matrix: np.ndarray, n_assets: int) -> np.ndarray:
    """
    --> BEST USED FOR BATCHES // Wrapperfunction for np.apply_along_axis
    Wrapper for the batch version, takes a concat vector of [action, flattened_covariance_matrix]
    the first n_action entries are the portfolio weights, the last n_action**2 entries represent a n_action x n_action covariance matrix
    """
    if np_action_flattened_covariance_matrix.ndim == 2: #batch case
        return np.apply_along_axis(
            calculate_estimated_portfolio_variance_from_action, 1,
            np_action_flattened_covariance_matrix,
            n_assets)
    else:
        portfolio_weight = np_action_flattened_covariance_matrix[:n_assets]
        tmp_covariance_matrix = np_action_flattened_covariance_matrix[n_assets:].reshape(n_assets, n_assets) #converts the flattened covariance back into a matrix

        est_portfolio_variance = calculate_portfolio_variance_from_covariance_matrix(np_portfolio_weight_vec=portfolio_weight,
                                                                                     np_covariance_matrix=tmp_covariance_matrix)
        return np.expand_dims(est_portfolio_variance,axis=0) #Since used with batches we want to return an narray, rather than a np.float64


def reconstruct_symmetric_matrix_from_triu(np_triu_without_diag: np.ndarray, np_diag: np.ndarray) -> np.ndarray:
    """
    from https://stackoverflow.com/questions/17527693/transform-the-upper-lower-triangular-part-of-a-symmetric-matrix-2d-array-into
    :param np_triu_without_diag:
    :param np_diag:
    :param size_dim:
    :return:
    """

    size_dim = np_diag.size

    reconstr_matrix = np.zeros((size_dim, size_dim))
    reconstr_matrix[np.triu_indices(reconstr_matrix.shape[0], k=1)] = np_triu_without_diag
    reconstr_matrix = reconstr_matrix + reconstr_matrix.T + np.diag(np_diag)

    return reconstr_matrix


def reconstruct_covariance_matrix_from_first_moment_and_second_moment_information(np_triu_without_diag: np.ndarray, np_diag: np.ndarray, np_mean_vec: np.ndarray) -> np.ndarray:
    """
    This function reconstruct the covariance matrix accoding to E[X^2]-E(X)^2 or in matrix notation:
    :param np_triu_without_diag:
    :param np_diag:
    :param np_mean_vec:
    """

    tmp_matrix_second_moment = reconstruct_symmetric_matrix_from_triu(np_triu_without_diag = np_triu_without_diag,
                                                                                      np_diag=np_diag)
    #Calculating the covariance matrix
    # Var(X)=E(XX^T)-m_x m_x^T
    est_covariance_matrix = tmp_matrix_second_moment - np.outer(np_mean_vec, np_mean_vec)
    #forces the diagonals to be strictly positive
    #This does however NOT enforce a semi-positive matrix as output, so it is not guaranteed that the matrix is a valid covariance matrix
    #-> Only important in the first few thousand initial epochs

    np.fill_diagonal(est_covariance_matrix, np.abs(np.diag(est_covariance_matrix)))
    return est_covariance_matrix

def reconstruct_flattened_covariance_matrix_from_first_moment_and_second_moment_information(
        flattened_first_moment_and_second_moment_information: np.ndarray,
        n_assets: int) -> np.ndarray:
    """
    --> BEST USED FOR BATCHES // Wrapperfunction for np.apply_along_axis
    Used to calculate a batch of covariance matricies from a batch of network outputs estimating first and second moments
    This is used for batch reconstruction where each row is decomposed as [mean_vec, triu_without_diag, diag]

    -> It is more convenient to return a 1d vector of n^2 x 1, i.e. we flatten the output matrix instead of returning a 2D matrix of n x n
    """

    if flattened_first_moment_and_second_moment_information.ndim == 2: #batch case
        return np.apply_along_axis(reconstruct_flattened_covariance_matrix_from_first_moment_and_second_moment_information, 1,
                            flattened_first_moment_and_second_moment_information,
                            n_assets)
    else:
        diag_size = n_assets
        triu_without_diag_size = int(n_assets*(n_assets-1)/2)

        tmp_mean_vec = flattened_first_moment_and_second_moment_information[:n_assets]

        tmp_flat_second_moment_matrix_triu_without_diag = flattened_first_moment_and_second_moment_information[n_assets:n_assets+triu_without_diag_size]
        tmp_flat_second_moment_matrix_diag = flattened_first_moment_and_second_moment_information[n_assets+triu_without_diag_size:n_assets+triu_without_diag_size+diag_size]

        tmp_reconstructed_covariance = reconstruct_covariance_matrix_from_first_moment_and_second_moment_information(np_triu_without_diag=tmp_flat_second_moment_matrix_triu_without_diag,
                                                                                        np_diag=tmp_flat_second_moment_matrix_diag,
                                                                                        np_mean_vec=tmp_mean_vec)
        return tmp_reconstructed_covariance.reshape(n_assets**2) #-->can be reconstructed by using .reshape(n_actions, n_actions)


def calculate_portfolio_variance_from_covariance_matrix(np_portfolio_weight_vec: np.ndarray, np_covariance_matrix: np.ndarray) -> np.float64:
    """
    For formula background check https://faculty.washington.edu/ezivot/econ424/portfolioTheoryMatrix.pdf
    :param covariance_matrix:
    :param portfolio_weight_vector:
    :return:
    """

    np_portfolio_variance = np.matmul(np_portfolio_weight_vec, np.matmul(np_covariance_matrix, np.transpose(np_portfolio_weight_vec)))
    return np.abs(np_portfolio_variance) #np.max(np_portfolio_variance, np.zeros_like(np_portfolio_variance)) #enforces variance to be >= /only relevant if the covariance matrix is not positive semidefinite, which happens in initialization phase of the training

def calculate_estimated_portfolio_variance_from_action(np_action_flattened_covariance_matrix: np.ndarray, n_assets: int) -> np.ndarray:
    """
    --> BEST USED FOR BATCHES // Wrapperfunction for np.apply_along_axis
    Wrapper for the batch version, takes a concat vector of [action, flattened_covariance_matrix]
    the first n_action entries are the portfolio weights, the last n_action**2 entries represent a n_action x n_action covariance matrix
    """
    if np_action_flattened_covariance_matrix.ndim == 2: #batch case
        return np.apply_along_axis(
            calculate_estimated_portfolio_variance_from_action, 1,
            np_action_flattened_covariance_matrix,
            n_assets)
    else:
        portfolio_weight = np_action_flattened_covariance_matrix[:n_assets]
        tmp_covariance_matrix = np_action_flattened_covariance_matrix[n_assets:].reshape(n_assets, n_assets) #converts the flattened covariance back into a matrix

        est_portfolio_variance = calculate_portfolio_variance_from_covariance_matrix(np_portfolio_weight_vec=portfolio_weight,
                                                                                     np_covariance_matrix=tmp_covariance_matrix)
        return np.expand_dims(est_portfolio_variance,axis=0) #Since used with batches we want to return an narray, rather than a np.float64


def construct_triu_from_symmetric_matrix(np_symmetric_matrix: np.ndarray) -> np.ndarray:
    """

    :param np_symmetric_matrix:
    :return:
    """
    second_moment_diag = np.diag(np_symmetric_matrix)
    second_moment_triu_without_diag_flat = np_symmetric_matrix[np.triu_indices(np_symmetric_matrix.shape[0], k=1)]

    return np.concatenate((second_moment_triu_without_diag_flat, second_moment_diag), axis=None)

def construct_first_moment_and_second_moment_information(np_observation: np.ndarray) -> np.ndarray:
    """
    Non-Batch case: Takes a a single N-dim vector R (i.e. a SINGLE OBSERVATION) and
    calculates the matrix RR^T which is NxN
    In a second step the symmetric NxN matrix is brought into a diagonal format:
    Due to being symmetric to store all contained information we just need the diagonal and all the entries above
    the diagonal (triangle upper/triu)
    :param self:
    :param row:
    :return: A Vector of dimension N + (N+1)*N/2
    """

    if np_observation.ndim == 2:  # Checks for batch calculation
        return np.apply_along_axis(construct_first_moment_and_second_moment_information, 1, arr=np_observation)
    else:
        np_single_observation = np_observation
        second_moment_mat = np.matmul(np.transpose(np.array([np_single_observation])), np.array([np_single_observation]))
        second_moment_triu = construct_triu_from_symmetric_matrix(second_moment_mat)
        return np.concatenate((np_single_observation, second_moment_triu), axis=None)

def torch_construct_first_moment_and_second_moment_information(torch_rewards: torch.Tensor) -> torch.Tensor:
    second_moment_mat = torch.matmul(torch.transpose(torch_rewards.unsqueeze(0), 0, 1), torch_rewards.unsqueeze(0))
    #print(second_moment_mat)
    #print(torch.triu_indices(*second_moment_mat.shape, offset=1))
    #print(second_moment_mat.index_select(1,torch.triu_indices(*second_moment_mat.shape, offset=1)))
    #torch.index_select(second_moment_mat, 1, torch.triu_indices(*second_moment_mat.shape, offset=1))

def calculate_portfolio_variance_from_action(flattened_covariance_matrix: np.ndarray, np_portfolio_weight: np.ndarray) -> np.ndarray:
    """
    --> BEST USED FOR BATCHES // Wrapperfunction for np.apply_along_axis
    Wrapper for the batch version, takes a concat vector of [action, flattened_covariance_matrix]
    the first n_action entries are the portfolio weights, the last n_action**2 entries represent a n_action x n_action covariance matrix
    """
    n_assets = np_portfolio_weight.shape[1]
    np_action_flattened_covariance_matrix = np.concatenate([np_portfolio_weight, flattened_covariance_matrix], axis=1)
    return calculate_estimated_portfolio_variance_from_action(
        np_action_flattened_covariance_matrix=np_action_flattened_covariance_matrix, n_assets=n_assets)


def calculate_estimated_portfolio_variance_from_action(np_action_flattened_covariance_matrix: np.ndarray, n_assets: int) -> np.ndarray:
    """
    --> BEST USED FOR BATCHES // Wrapperfunction for np.apply_along_axis
    Wrapper for the batch version, takes a concat vector of [action, flattened_covariance_matrix]
    the first n_action entries are the portfolio weights, the last n_action**2 entries represent a n_action x n_action covariance matrix
    """
    if np_action_flattened_covariance_matrix.ndim == 2: #batch case
        return np.apply_along_axis(
            calculate_estimated_portfolio_variance_from_action, 1,
            np_action_flattened_covariance_matrix,
            n_assets)
    else:
        portfolio_weight = np_action_flattened_covariance_matrix[:n_assets]
        tmp_covariance_matrix = np_action_flattened_covariance_matrix[n_assets:].reshape(n_assets, n_assets) #converts the flattened covariance back into a matrix

        est_portfolio_variance = calculate_portfolio_variance_from_covariance_matrix(np_portfolio_weight_vec=portfolio_weight,
                                                                                     np_covariance_matrix=tmp_covariance_matrix)
        return np.expand_dims(est_portfolio_variance,axis=0) #Since used with batches we want to return an narray, rather than a np.float64


def check_numpy_same_shape(array_1, array_2):
    shape_1 = array_1.shape
    shape_2 = array_2.shape
    list_bool = []
    if len(shape_1)==len(shape_2):
        for i, val_1 in enumerate(shape_1):
            list_bool.append(val_1 == shape_2[i])
        return all(list_bool)
    else:
        return False

def replay_episode_store_memory(gtrxl_model, torch_input):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #Torch input depends if you just use the obs as input or obs+action
    if not torch.is_tensor(torch_input):
        torch_input = torch.from_numpy(torch_input).float()
    initial_state = []
    initial_step = True
    input_dict={}
    cfg = gtrxl_model.config
    for i in range(cfg["attention_num_transformer_units"]):
        # The first 1 comes from the trajectory and is only valid in its batched form later we have to batch these
        initial_state.append(torch.zeros(1, cfg["attention_memory_inference"], cfg[
            "attention_dim"], device=device))

    #since we loop per timestep the seq list is always [1]
    seq_lens = torch.ones(1, dtype=torch.int32)

    for entry in torch_input:
        input_dict["obs"] = torch.unsqueeze(entry, dim=0)

        if initial_step:
            state = initial_state
            #print(input_dict["obs"].shape)
            #print(state[0].shape)
            #print(seq_lens.shape)
            _features, _features_2, memory_outs = gtrxl_model(input_dict, state, seq_lens)
            initial_step = False
        else:
            _features, _features_2, memory_outs = gtrxl_model(input_dict, state, seq_lens)

        #updating the state
        state = [torch.cat((state[i], torch.unsqueeze(memory_outs[i], 0)), 1)[:, 1:, :] for i in
                 range(cfg["attention_num_transformer_units"])]

    final_memory_state=state

    return final_memory_state


def reverse_one_hot_torch(tensor_one_hot, original_space):
    """
    Requires input tensor in batch format
    :param tensor_one_hot:
    :param original_space:
    :return:
    """
    if isinstance(original_space, Discrete):
        return torch.argmax(tensor_one_hot, dim=1).long()
        #return nn.functional.one_hot(x.long(), space.n)
    elif isinstance(original_space, MultiDiscrete):
        list_sub_discrete_space_elements = list(original_space.nvec)
        list_torch_split = torch.split(tensor_one_hot, list_sub_discrete_space_elements, dim=1)
        list_torch_argmax = [torch.argmax(split_tensor, dim=1) for split_tensor in list_torch_split]
        return torch.stack(list_torch_argmax, dim=1).long()
    else:
        raise ValueError("Unsupported space for `one_hot`: {}".format(original_space))

def reverse_one_hot_np(np_one_hot, original_space):
    """
    Requires input numpy array in batch format
    :param tensor_one_hot:
    :param original_space:
    :return:
    """
    if isinstance(original_space, Discrete):
        if np_one_hot.ndim==2:
            return np.argmax(np_one_hot, axis=1).astype(int)
        elif np_one_hot.ndim==1:
            return np.argmax(np_one_hot, axis=0).astype(int)
        #return nn.functional.one_hot(x.long(), space.n)
    elif isinstance(original_space, MultiDiscrete):
        list_sub_discrete_space_elements = list(original_space.nvec)
        list_numpy_split = np.split(np_one_hot, np.cumsum(list_sub_discrete_space_elements)[:-1], axis=1)
        #print(list_numpy_split)
        list_torch_argmax = [np.argmax(split_numpy, axis=1) for split_numpy in list_numpy_split]
        return np.stack(list_torch_argmax, axis=1).astype(int)
    else:
        raise ValueError("Unsupported space for `one_hot`: {}".format(original_space))

def estimate_standard_deviation(pred_moment_one: torch.Tensor, pred_moment_two: torch.Tensor) -> torch.Tensor:
    """
    Input are torch tensors
    :param pred_moment_one:
    :param pred_moment_two:
    :return:
    """
    torch_tmp_variance_estimate = pred_moment_two.flatten() - torch.pow(pred_moment_one.flatten(), 2.0)

    torch_tmp_variance_estimate = torch_tmp_variance_estimate.detach()  # to ensure that no gradient is
    torch_tmp_standard_deviation_estimate = torch.sqrt(
        torch.maximum(torch_tmp_variance_estimate, torch.zeros_like(torch_tmp_variance_estimate))
    )

    return torch_tmp_standard_deviation_estimate

def generate_allocation_bar_chart(np_allocation_weights, ymax=2.0, ymin=-1.0):
    fig, ax = plt.subplots(figsize=(10, 4))
    # ax = fig.add_axes([0,0,1,1])
    # langs = ['C', 'C++', 'Java', 'Python', 'PHP']
    #allocation = np.array(
    #    [0.33333, 0.4, -0.2, 0.1, 2.2, 0.33333, 0.4, -0.2, 0.1, 2.2, 0.33333, 0.4, -0.2, 0.1, 2.2])  # [23,17,35,29,12]
    list_labels = [f'a_{entry}' for entry in range(np_allocation_weights.size)]

    plt.bar(list_labels, np_allocation_weights,  # color ='maroon',
            width=0.4)
    plt.axhline(y=0, xmin=-100, xmax=100, color='black')
    # for a,b in zip(list_labels, allocation.tolist()):
    #    plt.text(a, b, str(b))
    for i, v in enumerate(np_allocation_weights.tolist()):
        plt.text(i - .155, v + .1, f'{v:.2f}', color='black')  # , fontweight='bold')

    plt.ylim(-1, 2)
    plt.grid(True)
    return plt
    # plt.xlim(0,10)
    # ax.bar(langs,students)
    #plt.show()

##### Constraint related functions
def convert_mask_into_index_set(list_mask):
    tmp_flatten =np.argwhere(np.array(list_mask) == 1).flatten()
    tmp_set = set(tmp_flatten.tolist())
    return tmp_set

def enrich_constraint_tuple_by_other_constraint_relations(list_tuple_setbased):
    """
    Compares every constraint to all the other constraints and determines the relationship, i.e. are the indices
    subsets to each other or not:
    H means subset, i.e. equal or smaller
    G means non subset
    This applies the set filter logic
    :param list_tuple_setbased:
    :return:
    """
    list_subset_filtered = []
    for tuple_k in list_tuple_setbased:
        tmp_dict = {
            "H+": [],
            "H-": [],
            "G+":[],
            "G-":[]
                    }
        for tuple_j in list_tuple_setbased:
            if(len(tuple_k[0].intersection(tuple_j[0]))>0): #no empty set
                if tuple_j[0].issubset(tuple_k[0]):
                    if tuple_j[2]=="+":
                        tmp_dict.get("H+").append((tuple_k[0].intersection(tuple_j[0]), tuple_j[3], tuple_j[1]))
                    if tuple_j[2]=="-":
                        tmp_dict.get("H-").append((tuple_k[0].intersection(tuple_j[0]), tuple_j[3], tuple_j[1]))
                else: # the G case that tuple_j must be bigger than tuple_k
                    if tuple_j[2]=="+":
                        tmp_dict.get("G+").append((tuple_k[0].intersection(tuple_j[0]), tuple_j[3], tuple_j[1]))
                    if tuple_j[2]=="-":
                        tmp_dict.get("G-").append((tuple_k[0].intersection(tuple_j[0]), tuple_j[3], tuple_j[1]))
        list_subset_filtered.append(tmp_dict)

    enriched_constraint_set = []
    #Adding for each constraint the G/H's (including the own constraint)
    for idx, subset_filtered in enumerate(list_subset_filtered):
        #adding a value to the tuple
        enriched_constraint_set.append(list_tuple_setbased[idx]+(subset_filtered,))

    return enriched_constraint_set #list_subset_filtered

def generate_aggregated_constraints_conditional_minkowski_S4_U1(head_factor_list, action_mask_dict, full_constraint_check=False):
    """
    Generates aggregated constraints for the 2n case
    :return:
    """
    list_conditional_minkowski_encoding_constraint_tuples = list(action_mask_dict.values())

    amount_assets = len(list_conditional_minkowski_encoding_constraint_tuples[0])

    np_ones = np.ones(amount_assets)

    list_agg_constraints = []

    # V1 condition
    list_agg_constraints.append((np.array(list_conditional_minkowski_encoding_constraint_tuples[2]), head_factor_list[2],'>='))
    # V2 condition
    list_agg_constraints.append(
        (np.array(list_conditional_minkowski_encoding_constraint_tuples[3]), head_factor_list[3], '>='))

    if full_constraint_check:
        # sums up to one
        list_agg_constraints.append((np_ones, 1.0, '>='))
        list_agg_constraints.append((np_ones, 1.0, '<='))

        # single variable conditions
        list_np_single_variables = list(np.eye(amount_assets))
        for single_variable in list_np_single_variables:
            list_agg_constraints.append((single_variable, 0.0, '>='))

    return list_agg_constraints


def generate_aggregated_constraints_conditional_minkowski_encoding(head_factor_list, action_mask_dict,
                                                                   full_constraint_check=False,
                                                                   conditional_minkowski_encoding_type="S4_U1"):
    """
    Generates aggregated constraints for the 2n case
    :return:
    """
    if conditional_minkowski_encoding_type=="S4_U1":
        return generate_aggregated_constraints_conditional_minkowski_S4_U1(head_factor_list, action_mask_dict,
                                                                    full_constraint_check)
    elif conditional_minkowski_encoding_type == "S4" or conditional_minkowski_encoding_type == "S3" or conditional_minkowski_encoding_type == "S2":
        helper_env_config = {
            "head_factor_list": head_factor_list,
            "action_mask_dict": action_mask_dict
        }
        list_raw_constraint_tuples = generate_list_raw_constraint_tuples(helper_env_config)
        list_relationship_enriched_tuple = convert_raw_constraints_to_full_constraint_tuple(
            list_constraint_tuple=list_raw_constraint_tuples)

        list_agg_constraints = generate_aggregated_constraints(list_relationship_enriched_tuple)

        if full_constraint_check:
            number_assets=list_agg_constraints[0][0].size
            list_min_max_boundaries = generate_min_max_boundaries(number_assets)
            list_agg_constraints.extend(list_min_max_boundaries)

        return list_agg_constraints

    else:
        raise ValueError(f'Unknown encoding type {conditional_minkowski_encoding_type}')


def generate_min_max_boundaries(number_assets):
    list_min_max_boundaries = []
    # single variable conditions
    list_np_single_variables = list(np.eye(number_assets))
    for single_variable in list_np_single_variables:
        list_min_max_boundaries.append((single_variable, 0.0, '>='))

    return list_min_max_boundaries

def generate_aggregated_constraints(enriched_constraint_set):
    """
    Aggregates the aux variables back into variables to then generate the aggregated constraints
    :param enriched_constraint_set:
    :return:
    """
    list_constraints = []
    for constraint_entry in enriched_constraint_set:
        dict_intersections = constraint_entry[5]

        h_plus_sum = sum([tuple_entry[2] for tuple_entry in dict_intersections.get("H+")])
        h_minus_sum = sum([tuple_entry[2] for tuple_entry in dict_intersections.get("H-")])
        g_plus_sum = sum([tuple_entry[2] for tuple_entry in dict_intersections.get("G+")])
        g_minus_sum = sum([tuple_entry[2] for tuple_entry in dict_intersections.get("G-")])

        #print(h_plus_sum)
        #print(h_minus_sum)
        list_constraints.append((constraint_entry[4], h_plus_sum+h_minus_sum+g_plus_sum ,"<="))
        list_constraints.append((constraint_entry[4], h_plus_sum+h_minus_sum+g_minus_sum ,">="))

    return list_constraints


def convert_raw_constraints_to_full_constraint_tuple(list_constraint_tuple):
    """
    Creates a list of constraint tuples with (var_index_set, constraint_val, constraint_sign, constraint_index, var_index_mask)
    :param list_constraint_tuple: [(constraint_val, [list_mask_var]), (constraint_val, [list_mask_var]), ...]
    :return: list_of tuples
    """
    list_tuple_setbased = []
    for idx, constraint in enumerate(list_constraint_tuple):
        tmp_c = constraint[0]
        tmp_mask = np.array(constraint[1])
        tmp_set = convert_mask_into_index_set(constraint[1])
        if tmp_c>=0:
            tmp_sign = "+"
        else:
            tmp_sign = "-"
        tmp_j = idx
        list_tuple_setbased.append((tmp_set, tmp_c, tmp_sign, tmp_j, tmp_mask))
        #(var_index_set, constraint_val, constraint_sign, constraint_index, var_index_mask)

    #enriching the constraints by additional info
    list_enriched_tuple = enrich_constraint_tuple_by_other_constraint_relations(list_tuple_setbased)
    # (var_index_set, constraint_val, constraint_sign, constraint_index, var_index_mask, dict_relationships)


    #return list_tuple_setbased
    return list_enriched_tuple

def generate_penalty_vec_from_samples_for_agg_constraint_satisfaction(np_samples, list_agg_constraints, correct_close_to_error=True, log_violations=True):
    #depreciated
    list_penalty_vectors = []
    for agg_constraint in list_agg_constraints:
        tmp_mask = agg_constraint[0]
        constraint_val = agg_constraint[1]
        constraint_type = agg_constraint[2]
        amount_samples = np_samples.shape[0]
        tmp_mask = np.tile(tmp_mask, (amount_samples, 1))
        masked_samples = tmp_mask*np_samples
        sum_samples = np.sum(masked_samples, axis=1)
        constraint_vector = np.ones_like(sum_samples)*constraint_val
        delta_vector = sum_samples-constraint_vector

        #Violations should result in positive penalty values
        if constraint_type=="<=":
            delta_vector = sum_samples - constraint_vector
            penalty_vector = delta_vector
            #penalty_vector = np.maximum(0.0, delta_vector)
        if constraint_type==">=":
            delta_vector = sum_samples - constraint_vector
            penalty_vector = delta_vector
            #penalty_vector = np.minimum(0.0, delta_vector)

        abs_penalty_vector = np.abs(penalty_vector)
        if correct_close_to_error:
            zero_vector = np.zeros_like(abs_penalty_vector)
            tmp_bool_violation = np.isclose(zero_vector, abs_penalty_vector, atol=1e-06) # deviations of less than 0.000001 are ignored
            abs_penalty_vector = np.where(tmp_bool_violation, 0.0, abs_penalty_vector) #replaces close to
            # zero values by zero
            if log_violations:
                write_constraint_violations(tuple_violated_agg_constraint=agg_constraint,
                                            abs_penalty_vector=abs_penalty_vector,
                                            np_samples=np_samples)

        list_penalty_vectors.append(abs_penalty_vector)

    return list_penalty_vectors

def write_constraint_violations(path, np_violating_action, np_violation_constraint_index):
    output_file = 'violation_history.csv'
    full_path_incl_file = f'{path}/{output_file}'

    try: # first time error check
        df_read_only = pd.read_csv(full_path_incl_file)
        #to avoid bloating just get active if the entries are not too present
        amount_tracked_violations = len(df_read_only.index)
    except FileNotFoundError:
        amount_tracked_violations = 0

    prob_factor = 1.0 / float(max(1, amount_tracked_violations - 100))  # first 100 observations should not be penalized

    # only execute if we do not have too many entries already
    if (np.random.rand() < prob_factor):

        #filter out the dummies that dont have a single violation, i.e. no -1
        #dummy_filter_mask = np.any(np_violating_action < 0)
        #print(dummy_filter_mask)

        # again sampling since we are passed an entire batch
        amount_max_random_samples = 2
        amount_max_random_samples = min(np_violating_action.shape[0], amount_max_random_samples)
        indices_to_write = np.random.choice(np.arange(np_violating_action.shape[0]), amount_max_random_samples, replace=False)

        #applying sample mask
        np_violation_constraint_index = np_violation_constraint_index[indices_to_write, :]
        np_violating_action = np_violating_action[indices_to_write, :]

        # merging the columns to a single string
        df_constraint_index = pd.DataFrame(np_violation_constraint_index).applymap(str).T.agg('_'.join).to_frame()

        df = pd.DataFrame(np_violating_action, columns=[f'asset_{i}' for i in range(np_violating_action.shape[1])])
        df["const_violations"] = df_constraint_index.iloc[:, 0]
        # print(os.getcwd())
        # print(output_path)
        df.to_csv(full_path_incl_file, mode='a', header=not os.path.exists(full_path_incl_file))

def filter_out_dummy_constraint_violations(np_violation_samples, np_index_violation_mask):

    dummy_filter_mask = np.any(np_index_violation_mask >= 0, axis=1) #-1 means no violation

    np_violation_samples_filtered = np_violation_samples[dummy_filter_mask, :]
    np_index_violation_mask_filtered = np_index_violation_mask[dummy_filter_mask, :]

    if np_violation_samples_filtered.size > 0:
        amount_to_max_sample = 2
        amount_to_sample = min(amount_to_max_sample, np_violation_samples_filtered.shape[0])
        np_selected_violation_indices = np.random.choice(np.arange(np_violation_samples_filtered.shape[0]), amount_to_sample, replace=False)
        # applying sample mask
        np_violation_samples_filtered = np_violation_samples_filtered[np_selected_violation_indices, :]
        np_index_violation_mask_filtered = np_index_violation_mask_filtered[np_selected_violation_indices, :]
        return np_violation_samples_filtered, np_index_violation_mask_filtered
    else:
        return np_violation_samples_filtered, np_index_violation_mask_filtered

    #reduce dataset to avoid bloating the entries too much
    # amount_to_sample = min(amount_to_sample, np_array_violating_index.shape[0])
    # np_selected_violation_indices = np.random.choice(np_array_violating_index, amount_to_sample, replace=False)

    # A[np.random.choice(np_array_violating_index, 2, replace=False), :]
    # A[np.random.choice(np.n.shape[0], 2, replace=False), :]



def log_constraint_violations(sample_batch, tmp_bool_matrix, action_processed):

    #np_samples = sample_batch[SampleBatch.ACTIONS]

    #generate np with the violation indices for rows
    #np_array_violating_index = np.arange(any_violation_in_allocation.shape[0])[any_violation_in_allocation]
    np_index_mask = np.tile(np.arange(tmp_bool_matrix.shape[1]), (tmp_bool_matrix.shape[0], 1))
    np_index_mask = (np_index_mask + np.ones_like(np_index_mask)) * tmp_bool_matrix - np.ones_like(
        np_index_mask)  # problems due to zero being the inverse element of multiplcation

    sample_batch[Postprocessing_Custom.ACTION_VIOLATIONS] = action_processed #np_samples #This is logged anyway through the actions
    sample_batch[Postprocessing_Custom.ACTION_VIOLATIONS_CONSTRAINT_VIOLATIONS] = np_index_mask

    #amount_to_sample = 2
    #amount_to_sample = min(amount_to_sample, np_array_violating_index.shape[0])
    #np_selected_violation_indices = np.random.choice(np_array_violating_index, amount_to_sample, replace=False)

    #A[np.random.choice(np_array_violating_index, 2, replace=False), :]
    #A[np.random.choice(np.n.shape[0], 2, replace=False), :]
    """
    violating_samples = np_samples[np_selected_violation_indices, :]

    if violating_samples.size > 0:

        np_index_mask = np.tile(np.arange(tmp_bool_matrix.shape[1]), (tmp_bool_matrix.shape[0], 1))
        np_index_mask = (np_index_mask + np.ones_like(np_index_mask)) * tmp_bool_matrix - np.ones_like(
        np_index_mask)  # problems due to zero being the inverse element of multiplcation
        index_mask_violation_samples = np_index_mask[np_selected_violation_indices, :]

        sample_batch[Postprocessing_Custom.ACTION_VIOLATIONS] = violating_samples
        sample_batch[Postprocessing_Custom.ACTION_VIOLATIONS_CONSTRAINT_VIOLATIONS] = index_mask_violation_samples
    else: # for RL reasons we need to add some dummy variables
        print("HIT THE PROBLEM")
        np_index_mask_dummy = np.tile(np.arange(tmp_bool_matrix.shape[1]), (2, 1))
        #print(np_index_mask_dummy)
        np_violation_dummy = np_samples[np.array([0, 1]), :]
        #print(np_violation_dummy)
        #print(np_index_mask_dummy.shape)
        #print(np_violation_dummy.shape)
        sample_batch[Postprocessing_Custom.ACTION_VIOLATIONS] = np_violation_dummy
        sample_batch[Postprocessing_Custom.ACTION_VIOLATIONS_CONSTRAINT_VIOLATIONS] = np_index_mask_dummy
        #print("---")
        #print(index_mask_violation_samples.shape)
        #print(violating_samples.shape)
    """ or None

def log_constraint_violations_working(any_violation_in_allocation, sample_batch, tmp_bool_matrix):

    np_samples = sample_batch[SampleBatch.ACTIONS]

    #generate np with the violation indices for rows
    np_array_violating_index = np.arange(any_violation_in_allocation.shape[0])[any_violation_in_allocation]
    amount_to_sample = 2
    amount_to_sample = min(amount_to_sample, np_array_violating_index.shape[0])
    np_selected_violation_indices = np.random.choice(np_array_violating_index, amount_to_sample, replace=False)

    #A[np.random.choice(np_array_violating_index, 2, replace=False), :]
    #A[np.random.choice(np.n.shape[0], 2, replace=False), :]

    violating_samples = np_samples[np_selected_violation_indices, :]

    if violating_samples.size > 0:

        np_index_mask = np.tile(np.arange(tmp_bool_matrix.shape[1]), (tmp_bool_matrix.shape[0], 1))
        np_index_mask = (np_index_mask + np.ones_like(np_index_mask)) * tmp_bool_matrix - np.ones_like(
        np_index_mask)  # problems due to zero being the inverse element of multiplcation
        index_mask_violation_samples = np_index_mask[np_selected_violation_indices, :]

        sample_batch[Postprocessing_Custom.ACTION_VIOLATIONS] = violating_samples
        sample_batch[Postprocessing_Custom.ACTION_VIOLATIONS_CONSTRAINT_VIOLATIONS] = index_mask_violation_samples

"""
        #print(amount_tracked_violations)
        prob_factor = 1.0/float(max(1, amount_tracked_violations-10)) #first 10 observations should not be penalized
        #print(prob_factor)

        #only execute if we do not have too many entries already
        if (np.random.rand() < prob_factor):

            sample_batch[]
            # index of violations
            np_index_mask = np.tile(np.arange(tmp_bool_matrix.shape[1]), (tmp_bool_matrix.shape[0], 1))
            np_index_mask = (np_index_mask + np.ones_like(np_index_mask)) * tmp_bool_matrix - np.ones_like(
                np_index_mask)  # problems due to zero being the inverse element of multiplcation

            #merging the columns to a single string
            df_constraint_index = pd.DataFrame(np_index_mask).applymap(str).T.agg('_'.join).to_frame()
            #selecting only relevant indices
            df_constraint_index = df_constraint_index.iloc[np_selected_violation_indices.tolist()].reset_index()

            df = pd.DataFrame(violating_samples, columns=[f'asset_{i}' for i in range(violating_samples.shape[1])])
            df["const_violations"] = df_constraint_index.iloc[:, 1]
            #print(os.getcwd())
            #print(output_path)
            df.to_csv(output_path, mode='a')#, header=not os.path.exists(output_path))
        """ or None

def check_constraint_violations(np_penalty_matrix, sample_batch=None, action_processed=None):
    """
    Returns total amount of solutions which have at least one constraint valuation
    :param list_constraints:
    :param list_penalty_vectors:
    :return:
    """
    log_violations = True # for now set to true

    zero_matrix = np.zeros_like(np_penalty_matrix)

    #1. make all entries close to zero FALSE -> leave only potentially material deviations at True
    tmp_bool_matrix_non_zero = ~np.isclose(zero_matrix, np_penalty_matrix, atol=0.0005) #atol=0.0001)

    #2. make all positive penalty value entries True:
    tmp_bool_all_positive = np_penalty_matrix > 0 # this means there is a violation

    tmp_bool_matrix = tmp_bool_matrix_non_zero & tmp_bool_all_positive

    amount_violation_in_allocation = np.sum(tmp_bool_matrix, axis=1)


    #any_violation_in_allocation = np.any(tmp_bool_matrix, axis=1)

    if log_violations:
        log_constraint_violations(sample_batch=sample_batch,
                                  tmp_bool_matrix=tmp_bool_matrix,
                                  action_processed=action_processed)

    #amount_allocations_in_violation = np.sum(any_violation_in_allocation)

    return amount_violation_in_allocation
"""
def write_constraint_violations(tuple_violated_agg_constraint, abs_penalty_vector, np_samples):

    #TODO---refactor---
    zero_vector = np.zeros_like(abs_penalty_vector)
    tmp_bool_violation = ~np.isclose(zero_vector, abs_penalty_vector)
    #print(np_samples)
    #print(np_samples.shape)
    violating_samples = np_samples[tmp_bool_violation]
    if violating_samples.size > 0:
        df = pd.DataFrame(violating_samples, columns=[f'asset_{i}' for i in range(violating_samples.shape[1])])

        converted_tuple = (np.array_str(tuple_violated_agg_constraint[0]), str(tuple_violated_agg_constraint[1]), str(tuple_violated_agg_constraint[2]))
        str_constraint = '|'.join(converted_tuple)
        df["constraint_violation"] = str_constraint

        #This is written to the main folder and not to the respective experiment
        output_path = 'violation_history.csv'
        df.to_csv(output_path, mode='a', header=not os.path.exists(output_path))
""" or None

def generate_np_penalty_matrix(np_samples, list_agg_constraints,
                               correct_close_to_error=True, only_allow_positive_values=True):#False):

    list_penalty_vectors = []
    for agg_constraint in list_agg_constraints:
        tmp_mask = agg_constraint[0]
        constraint_val = agg_constraint[1]
        constraint_type = agg_constraint[2]
        amount_samples = np_samples.shape[0]

        tmp_mask = np.tile(tmp_mask, (amount_samples, 1))

        masked_samples = tmp_mask * np_samples
        sum_samples = np.sum(masked_samples, axis=1)  # we have the sum per sample
        constraint_vector = np.ones_like(sum_samples) * constraint_val
        #delta_vector = sum_samples - constraint_vector

        #Violations should result in positive penalty values
        if constraint_type=="<=":
            delta_vector = sum_samples - constraint_vector
            penalty_vector = delta_vector
        if constraint_type==">=":
            delta_vector = constraint_vector - sum_samples
            penalty_vector = delta_vector

        # This only says that the penalties can not have negative values (which might cause problems,
        # by summing up costs over different periods negative ones might cancel out etc..
        if only_allow_positive_values:
            penalty_vector = np.maximum(np.zeros_like(penalty_vector), penalty_vector)

        #abs_penalty_vector = np.abs(penalty_vector)

        if correct_close_to_error:
            zero_vector = np.zeros_like(penalty_vector)
            tmp_bool_violation = np.isclose(zero_vector, penalty_vector,
                                            atol=1e-05)  # deviations of less than 0.000001 are ignored
            #penalty_vector = np.where(tmp_bool_violation, 0.0, penalty_vector)  # replaces close to
            # zero values by zero

        list_penalty_vectors.append(penalty_vector)

    return np.transpose(np.array(list_penalty_vectors))

def generate_list_raw_constraint_tuples(env_config):

    if "head_factor_list" in env_config:
        list_head_factor = env_config.get("head_factor_list")
    if "action_mask_dict" in env_config:#env_config
        action_mask_dict = env_config.get("action_mask_dict")

    list_raw_constraint_tuples = []
    for idx, head_factor in enumerate(list_head_factor):
        tmp_list_action_mask = action_mask_dict.get(f'{idx}_action_mask')
        tmp_tuple = (list_head_factor[idx], tmp_list_action_mask)
        list_raw_constraint_tuples.append(tmp_tuple)
    return list_raw_constraint_tuples


def calculate_action_allocation(np_raw_action, env_config, model=None):
    """
    This is for np_raw_action (i.e. not in a dictionary form)
    Aggregates np_raw_action
    :param np_raw_action:
    :param env_config:
    :return:
    """
    force_single_simplex = env_config.get("force_single_simplex", False)
    force_box_space = env_config.get("force_box_space", False)
    force_single_simplex_scaling_dict = env_config.get("force_single_simplex_scaling_dict", False)

    actor_conditional_minkowski_encoding_type = env_config.get("actor_conditional_minkowski_encoding_type", None)

    if not force_single_simplex and not force_box_space and \
            not force_single_simplex_scaling_dict and "head_factor_list" in env_config:
        if actor_conditional_minkowski_encoding_type is None:
            list_head_factor = env_config.get("head_factor_list")
            if np_raw_action.ndim == 2:
                list_sub_actions = np.split(np_raw_action, len(list_head_factor), axis=1)
            else:
                list_sub_actions = np.split(np_raw_action, len(list_head_factor), axis=0)
            list_allocation = []
            for idx, head_factor in enumerate(list_head_factor):
                tmp_action = head_factor * list_sub_actions[idx]
                list_allocation.append(tmp_action)

            action_merged = np.sum(list_allocation, axis=0)
            return action_merged
        else:
            if actor_conditional_minkowski_encoding_type == "S4_U1":
                list_head_factor = env_config.get("head_factor_list")
                dict_action_mask = env_config.get("action_mask_dict")
                dict_uniform_factor = env_config.get("uniform_factor_dict")

                tmp_dict_actions = TorchAutoregressiveDirichletDistributionS4_U1.action_output_wrapper(model, np_raw_action)

                processed_action = convert_conditional_minkowski_encoding_to_action(
                    dict_raw_actions=tmp_dict_actions,
                    dict_action_mask=dict_action_mask,
                    head_factor_list=list_head_factor,
                    dict_uniform_factor=dict_uniform_factor,
                    encoding_type=actor_conditional_minkowski_encoding_type)

                return processed_action
            elif env_config.get("conditional_minkowski_encoding_type") == "S4":
                list_head_factor = env_config.get("head_factor_list")
                dict_action_mask = env_config.get("action_mask_dict")
                dict_uniform_factor = env_config.get("uniform_factor_dict")

                tmp_dict_actions = TorchAutoregressiveDirichletDistributionS4.action_output_wrapper(model, np_raw_action)

                processed_action = convert_conditional_minkowski_encoding_to_action(
                    dict_raw_actions=tmp_dict_actions,
                    dict_action_mask=dict_action_mask,
                    head_factor_list=list_head_factor,
                    dict_uniform_factor=dict_uniform_factor,
                    encoding_type=env_config.get("conditional_minkowski_encoding_type"))
                return processed_action
            elif env_config.get("conditional_minkowski_encoding_type") == "S3":
                raise NotImplementedError
            elif env_config.get("conditional_minkowski_encoding_type") == "S2":
                raise NotImplementedError
            else:
                raise ValueError("Unknown encoding type")

    elif force_single_simplex_scaling_dict:
        fixed_scaling_factor = 1.3#None
        if np_raw_action.ndim == 2:
            list_sub_actions = np.split(np_raw_action, [np_raw_action.shape[1]-1, np_raw_action.shape[1]], axis=1)
            np_encoded_action = list_sub_actions[0]
            if fixed_scaling_factor is not None:
                np_scaling_factor_long = np.ones_like(list_sub_actions[1])*fixed_scaling_factor
            else:
                np_scaling_factor_long = list_sub_actions[1]
            np_scaling_factor_short = -(np_scaling_factor_long-1)

            #All weights need to be scaled by the same factor
            np_scaling_factor_total = np.abs(np_scaling_factor_long)+np.abs(np_scaling_factor_short)

            np_index_mask_0 = np.tile(np.array(env_config.get("action_mask_dict").get("0_action_mask")),
                                      (np_encoded_action.shape[0], 1))
            np_index_mask_1 = np.tile(np.array(env_config.get("action_mask_dict").get("1_action_mask")),
                                      (np_encoded_action.shape[0], 1))

            long_action = np_encoded_action*np_index_mask_0*np_scaling_factor_total
            short_action = np_encoded_action*np_index_mask_1*np_scaling_factor_total

            action_merged = long_action - short_action

            return action_merged
        else:
            #print("single")
            #print(np_raw_action)
            list_sub_actions = np.split(np_raw_action, [np_raw_action.shape[0] - 1, np_raw_action.shape[0]], axis=0)
            #print(list_sub_actions)
            #print("~~")
            #list_sub_actions = np.split(np_raw_action, len(list_head_factor), axis=1)
            np_encoded_action = list_sub_actions[0]
            if fixed_scaling_factor is not None:
                np_scaling_factor_long = np.ones_like(list_sub_actions[1]) * fixed_scaling_factor
            else:
                np_scaling_factor_long = list_sub_actions[1]
            np_scaling_factor_short = -(np_scaling_factor_long - 1)

            # All weights need to be scaled by the same factor
            np_scaling_factor_total = np.abs(np_scaling_factor_long) + np.abs(np_scaling_factor_short)

            np_scaling_factor_long_matrix = np.tile(np_scaling_factor_long,
                                                    (1, np_encoded_action.shape[0]))
            np_scaling_factor_short_matrix = np.tile(np_scaling_factor_short,
                                                     (1, np_encoded_action.shape[0]))

            # print(np_scaling_factor_long_matrix)
            # print(np_scaling_factor_short_matrix)
            # print("$$$")
            # print(np_encoded_action)
            # print(np_scaling_factor_long)
            # print(np_scaling_factor_short)
            # print("####")
            np_index_mask_0 = np.array(env_config.get("action_mask_dict").get("0_action_mask"))
            np_index_mask_1 = np.array(env_config.get("action_mask_dict").get("1_action_mask"))

            long_action = np_encoded_action * np_index_mask_0 * np_scaling_factor_total# * np_scaling_factor_long_matrix
            short_action = np_encoded_action * np_index_mask_1 * np_scaling_factor_total# * np_scaling_factor_short_matrix

            action_merged = long_action - short_action
            return action_merged
    elif force_single_simplex:
        return np_raw_action
    else:
        return np_raw_action

def calculate_action_dim(action_space):

    if isinstance(action_space, gym.spaces.Dict):
        space_output_dim_total = 0
        for space_name, space in action_space.spaces.items():
            space_output_dim = None
            if isinstance(space, gym.spaces.Discrete):
                space_output_dim = 1
            elif (
                    isinstance(space, gym.spaces.MultiDiscrete)
                    and space is not None
            ):
                space_output_dim = int(np.prod(space.shape))
            elif (isinstance(space, gym.spaces.Box)
                  and space is not None
            ):
                space_output_dim = 2*int(np.sum(space.shape))  # only valid for one dimensional .Box
            elif (isinstance(space, Simplex)
                  and space is not None
            ):
                space_output_dim = int(np.sum(space.shape))
            else:
                raise ValueError(f'Unknown space type {space}')
            space_output_dim_total+=space_output_dim
    else:
        space = action_space
        space_output_dim_total = 0
        if isinstance(space, gym.spaces.Discrete):
            space_output_dim = 1
        elif (
                isinstance(space, gym.spaces.MultiDiscrete)
                and space is not None
        ):
            space_output_dim = int(np.prod(space.shape))
        elif (isinstance(space, gym.spaces.Box)
              and space is not None
        ):
            space_output_dim = 2*int(np.sum(space.shape))  # only valid for one dimensional .Box #*2 because of two parameters, mean+var for normal distribution
        elif (isinstance(space, Simplex)
              and space is not None
        ):
            space_output_dim = int(np.sum(space.shape))
        else:
            raise ValueError(f'Unknown space type {space}')
        space_output_dim_total += space_output_dim

    return space_output_dim_total


def convert_conditional_minkowski_encoding_to_true_greater_equal_constraints():
    raise NotImplementedError

def convert_conditional_minkowski_encoding_to_action(dict_raw_actions, dict_action_mask, head_factor_list, dict_uniform_factor=None, encoding_type="S4"):

    if encoding_type=="S4": # this is typically the basecase and covers multiple setups
        return decode_S4(dict_raw_actions, dict_action_mask, head_factor_list)
    elif encoding_type=="S4_U1":
        return decode_S4_U1(dict_raw_actions, dict_action_mask, head_factor_list, dict_uniform_factor)


def decode_S4_U1(dict_raw_actions, dict_action_mask, head_factor_list, dict_uniform_factor):

    mask_S1 = np.array(dict_action_mask.get("0_action_mask"), dtype=bool)
    mask_S2 = np.array(dict_action_mask.get("1_action_mask"), dtype=bool)
    mask_S3 = np.array(dict_action_mask.get("2_action_mask"), dtype=bool)
    mask_S4 = np.array(dict_action_mask.get("3_action_mask"), dtype=bool)

    mask_Q2 = np.logical_and(mask_S3, mask_S4)
    mask_Q1 = np.logical_xor(mask_Q2, mask_S3)
    mask_Q3 = np.logical_xor(mask_Q2, mask_S4)
    mask_Q4 = np.logical_not(np.logical_or(mask_S3, mask_S4))

    scaling_factor_max_U1 = dict_uniform_factor.get("0_uniform_factor")

    # sf_1
    c_1 = head_factor_list[2]
    c_2 = head_factor_list[3]
    if dict_raw_actions.get("0_allocation").ndim == 2:
        batch_size = dict_raw_actions.get("0_allocation").shape[0]

        #sampled scaling factor -> we get a [0,1] action for 0_uniform_factor and have to scale it down accordingly
        c_Q4 = np.squeeze(dict_raw_actions.get("0_uniform_factor"))*scaling_factor_max_U1
        scaling_factor_1 = c_Q4
        scaled_action_1 = np.multiply(dict_raw_actions.get("0_allocation"), scaling_factor_1[:, np.newaxis])

        # sf_2
        # c_1+ c_2 -1 +c_Q4
        scaling_factor_2 = c_1 + c_2 - np.ones(batch_size) + c_Q4
        # sf_3
        # 1-c_2-c_Q4
        scaling_factor_3 = np.ones(batch_size) - c_2 - c_Q4
        # sf_4
        # 1-c_1-c_Q4
        scaling_factor_4 = np.ones(batch_size) - c_1 - c_Q4

        scaled_action_2 = np.multiply(dict_raw_actions.get("1_allocation"), scaling_factor_2[:, np.newaxis])
        scaled_action_3 = np.multiply(dict_raw_actions.get("2_allocation"), scaling_factor_3[:, np.newaxis])
        scaled_action_4 = np.multiply(dict_raw_actions.get("3_allocation"), scaling_factor_4[:, np.newaxis])

        merged_action = scaled_action_1 + scaled_action_2 + scaled_action_3 + scaled_action_4
        return merged_action
    elif dict_raw_actions.get("0_allocation").ndim == 1:
        c_Q4 = np.squeeze(dict_raw_actions.get("0_uniform_factor"))*scaling_factor_max_U1
        scaling_factor_1 = c_Q4
        # https://stackoverflow.com/questions/22934219/numpy-multiply-arrays-rowwise
        scaled_action_1 = dict_raw_actions.get("0_allocation") * scaling_factor_1

        # sf_2
        # c_1+ c_2 -1 +c_Q4
        scaling_factor_2 = c_1 + c_2 - 1 + c_Q4
        # sf_3
        # 1-c_2-c_Q4
        scaling_factor_3 = 1 - c_2 - c_Q4
        # sf_4
        # 1-c_1-c_Q4
        scaling_factor_4 = 1 - c_1 - c_Q4

        scaled_action_2 = dict_raw_actions.get("1_allocation") * scaling_factor_2
        scaled_action_3 = dict_raw_actions.get("2_allocation") * scaling_factor_3
        scaled_action_4 = dict_raw_actions.get("3_allocation") * scaling_factor_4

        merged_action = scaled_action_1 + scaled_action_2 + scaled_action_3 + scaled_action_4
        return merged_action
    else:
        raise ValueError('Unknown amount of dimensions')

def decode_S4(dict_raw_actions, dict_action_mask, head_factor_list):
    mask_S1 = np.array(dict_action_mask.get("0_action_mask"), dtype=bool)
    mask_S2 = np.array(dict_action_mask.get("1_action_mask"), dtype=bool)
    mask_S3 = np.array(dict_action_mask.get("2_action_mask"), dtype=bool)
    mask_S4 = np.array(dict_action_mask.get("3_action_mask"), dtype=bool)

    mask_Q2 = np.logical_and(mask_S3, mask_S4)
    mask_Q1 = np.logical_xor(mask_Q2, mask_S3)
    mask_Q3 = np.logical_xor(mask_Q2, mask_S4)
    mask_Q4 = np.logical_not(np.logical_or(mask_S3, mask_S4))

    # sf_1
    c_1 = head_factor_list[2]
    c_2 = head_factor_list[3]

    if dict_raw_actions.get("0_allocation").ndim == 2:
        batch_size = dict_raw_actions.get("0_allocation").shape[0]

        mask_Q2 = np.tile(mask_Q2, (batch_size, 1))
        mask_Q1 = np.tile(mask_Q1, (batch_size, 1))
        mask_Q3 = np.tile(mask_Q3, (batch_size, 1))
        mask_Q4 = np.tile(mask_Q4, (batch_size, 1))
        c_1 = np.ones(batch_size) * c_1
        c_2 = np.ones(batch_size) * c_2

        # 1-max(c_1,c_2)
        scaling_factor_1 = (np.ones(batch_size) - np.amax(np.vstack((c_1, c_2)), axis=0))

        # https://stackoverflow.com/questions/22934219/numpy-multiply-arrays-rowwise
        scaled_action_1 = np.multiply(dict_raw_actions.get("0_allocation"), scaling_factor_1[:, np.newaxis])  #
        # raw_action_1 = dict_raw_actions.get("action_1")*scaling_factor_1
        c_Q1 = np.sum(scaled_action_1 * mask_Q1, axis=1)
        c_Q2 = np.sum(scaled_action_1 * mask_Q2, axis=1)
        c_Q3 = np.sum(scaled_action_1 * mask_Q3, axis=1)
        c_Q4 = np.sum(scaled_action_1 * mask_Q4, axis=1)

        # sf_2
        # c_1+ c_2 -1 +c_Q4 - c_Q2
        scaling_factor_2 = c_1 + c_2 - np.ones(batch_size) + c_Q4 - c_Q2
        # sf_3
        # 1-c_2-c_Q4-c_Q1
        scaling_factor_3 = np.ones(batch_size) - c_2 - c_Q4 - c_Q1
        # sf_4
        # 1-c_1-c_Q4-c_Q3
        scaling_factor_4 = np.ones(batch_size) - c_1 - c_Q4 - c_Q3

        scaled_action_2 = np.multiply(dict_raw_actions.get("1_allocation"), scaling_factor_2[:, np.newaxis])
        scaled_action_3 = np.multiply(dict_raw_actions.get("2_allocation"), scaling_factor_3[:, np.newaxis])
        scaled_action_4 = np.multiply(dict_raw_actions.get("3_allocation"), scaling_factor_4[:, np.newaxis])

        merged_action = scaled_action_1 + scaled_action_2 + scaled_action_3 + scaled_action_4
        return merged_action
    elif dict_raw_actions.get("0_allocation").ndim == 1:
        # 1-max(c_1,c_2)
        scaling_factor_1 = 1 - max(c_1, c_2)
        # https://stackoverflow.com/questions/22934219/numpy-multiply-arrays-rowwise
        scaled_action_1 = dict_raw_actions.get("0_allocation") * scaling_factor_1
        c_Q1 = np.sum(scaled_action_1 * mask_Q1, axis=0)
        c_Q2 = np.sum(scaled_action_1 * mask_Q2, axis=0)
        c_Q3 = np.sum(scaled_action_1 * mask_Q3, axis=0)
        c_Q4 = np.sum(scaled_action_1 * mask_Q4, axis=0)

        # sf_2
        # c_1+ c_2 -1 +c_Q4 - c_Q2
        scaling_factor_2 = c_1 + c_2 - 1 + c_Q4 - c_Q2
        # sf_3
        # 1-c_2-c_Q4-c_Q1
        scaling_factor_3 = 1 - c_2 - c_Q4 - c_Q1
        # sf_4
        # 1-c_1-c_Q4-c_Q3
        scaling_factor_4 = 1 - c_1 - c_Q4 - c_Q3

        scaled_action_2 = dict_raw_actions.get("1_allocation") * scaling_factor_2
        scaled_action_3 = dict_raw_actions.get("2_allocation") * scaling_factor_3
        scaled_action_4 = dict_raw_actions.get("3_allocation") * scaling_factor_4

        merged_action = scaled_action_1 + scaled_action_2 + scaled_action_3 + scaled_action_4
        return merged_action
    else:
        raise ValueError('Unknown amount of dimensions')