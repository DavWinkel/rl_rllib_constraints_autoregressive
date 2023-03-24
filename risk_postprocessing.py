import torch
from ray.rllib.policy.sample_batch import SampleBatch
from helper_functions import convert_raw_constraints_to_full_constraint_tuple, generate_aggregated_constraints,\
    generate_np_penalty_matrix, generate_list_raw_constraint_tuples, check_constraint_violations, \
    calculate_action_allocation, generate_aggregated_constraints_conditional_minkowski_encoding, \
    reconstruct_flattened_covariance_matrix_from_first_moment_and_second_moment_information, \
    calculate_estimated_portfolio_variance_from_action, construct_econ_returns, generate_trainable_dataset

from custom_keys import Postprocessing_Custom, SampleBatch_Custom
import numpy as np
from typing import Dict, Optional

from ray.rllib.evaluation.episode import Episode
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import AgentID

from ray.rllib.evaluation.postprocessing import discount_cumsum

from environment_wrapper import extract_environment_class_from_config


def calculate_penalty_score_short_ppo(policy, sample_batch):

    if not "tmp_experiment_path" in policy.config:
        import os
        policy.config["tmp_experiment_path"] = os.getcwd()

    list_raw_constraint_tuples = generate_list_raw_constraint_tuples(policy.config.get("env_config"))

    list_relationship_enriched_tuple = convert_raw_constraints_to_full_constraint_tuple(
        list_constraint_tuple=list_raw_constraint_tuples)
    list_agg_constraints = generate_aggregated_constraints(list_relationship_enriched_tuple)

    env_config = policy.config.get("env_config")

    # Merging only here necessary, since this is the only approach that is capable of applying the multi layer model
    action_processed = calculate_action_allocation(sample_batch[SampleBatch.ACTIONS], env_config)

    if isinstance(action_processed, torch.Tensor):
        is_input_tensor = True
    else:
        is_input_tensor = False

    if is_input_tensor:
        np_penalty_matrix = generate_np_penalty_matrix(action_processed,
                                                       list_agg_constraints, only_allow_positive_values=True)
        pass # TODO BE IMPLEMENTED
        raise NotImplementedError

    else:
        np_penalty_matrix = generate_np_penalty_matrix(action_processed,
                                                       list_agg_constraints, only_allow_positive_values=True)

    amount_constraint_violations = check_constraint_violations(np_penalty_matrix,
                                                               sample_batch=sample_batch,
                                                               action_processed=action_processed)

    unweighted_penalty_violation_score = np.sum(np.maximum(np.zeros_like(np_penalty_matrix), np_penalty_matrix), axis=1)

    return np_penalty_matrix, unweighted_penalty_violation_score, amount_constraint_violations


def compute_penalty_values_short_ppo(policy, sample_batch, other_agent_batches, episode):

    if Postprocessing_Custom.REWARD_BASE_NO_PENALTIES not in sample_batch:
        sample_batch[Postprocessing_Custom.REWARD_BASE_NO_PENALTIES] = sample_batch[SampleBatch.REWARDS]

    np_penalty_matrix, np_unweighted_penalty_violation_score, amount_constraint_violations = calculate_penalty_score_short_ppo(
        sample_batch=sample_batch,
        policy=policy)


    #sample_batch[Postprocessing_Custom.LOG_BARRIER_CONSTRAINT_PENALTIES] = np_weighted_processed_penalty_value
    sample_batch[Postprocessing_Custom.UNWEIGHTED_PENALTY_VIOLATION_SCORE] = np_unweighted_penalty_violation_score
    sample_batch[Postprocessing_Custom.AMOUNT_CONSTRAINT_VIOLATIONS] = amount_constraint_violations
    sample_batch[Postprocessing_Custom.BOOL_ANY_CONSTRAINT_VIOLATIONS] = (amount_constraint_violations>0).astype(int)

    return sample_batch


def compute_penalty_values_risk_baseline(policy, sample_batch, other_agent_batches, episode):
    if Postprocessing_Custom.REWARD_BASE_NO_PENALTIES not in sample_batch:
        sample_batch[Postprocessing_Custom.REWARD_BASE_NO_PENALTIES] = sample_batch[SampleBatch.REWARDS]

    np_penalty_matrix, np_unweighted_penalty_violation_score, amount_constraint_violations = calculate_penalty_score_risk_baseline(
        sample_batch=sample_batch,
        policy=policy)

    sample_batch[Postprocessing_Custom.UNWEIGHTED_PENALTY_VIOLATION_SCORE] = np_unweighted_penalty_violation_score
    sample_batch[Postprocessing_Custom.AMOUNT_CONSTRAINT_VIOLATIONS] = amount_constraint_violations
    sample_batch[Postprocessing_Custom.BOOL_ANY_CONSTRAINT_VIOLATIONS] = (amount_constraint_violations > 0).astype(int)

    return sample_batch


def calculate_penalty_score_risk_baseline(policy, sample_batch):

    env_config = policy.config.get("env_config")
    conditional_minkowski_encoding_type = policy.config.get("env_config").get("constraints_conditional_minkowski_encoding_type", None)


    if conditional_minkowski_encoding_type is not None:
        head_factor_list = policy.config.get("env_config").get("head_factor_list")
        action_mask_dict = policy.config.get("env_config").get("action_mask_dict")
        list_agg_constraints = generate_aggregated_constraints_conditional_minkowski_encoding(
            head_factor_list,
            action_mask_dict,
            full_constraint_check=True,
            conditional_minkowski_encoding_type=conditional_minkowski_encoding_type)
    else:
        #standard procedure
        list_raw_constraint_tuples = generate_list_raw_constraint_tuples(policy.config.get("env_config"))

        list_relationship_enriched_tuple = convert_raw_constraints_to_full_constraint_tuple(
            list_constraint_tuple=list_raw_constraint_tuples)
        list_agg_constraints = generate_aggregated_constraints(list_relationship_enriched_tuple)

    #2) Generate correct processed action:
    action_processed = calculate_action_allocation(sample_batch[SampleBatch.ACTIONS], env_config, policy.model)

    if isinstance(sample_batch[SampleBatch.ACTIONS], torch.Tensor):
        is_input_tensor = True
    else:
        is_input_tensor = False

    if is_input_tensor:
        np_penalty_matrix = generate_np_penalty_matrix(action_processed,
                                                       list_agg_constraints, only_allow_positive_values=True)
        pass # TODO BE IMPLEMENTED
        raise NotImplementedError

    else:
        np_penalty_matrix = generate_np_penalty_matrix(action_processed,
                                                       list_agg_constraints, only_allow_positive_values=True)

    amount_constraint_violations = check_constraint_violations(np_penalty_matrix,
                                                               sample_batch=sample_batch,
                                                               action_processed=action_processed)

    unweighted_penalty_violation_score = np.sum(np.maximum(np.zeros_like(np_penalty_matrix), np_penalty_matrix), axis=1)

    # HERE WE OVERWRITE THE REWARD # No adjustment needed in this case
    sample_batch[SampleBatch.REWARDS] = sample_batch[SampleBatch.REWARDS]

    return np_penalty_matrix, unweighted_penalty_violation_score, amount_constraint_violations


def compute_penalty_values_risk_autoregressive_ppo(policy, sample_batch, other_agent_batches, episode):
    if Postprocessing_Custom.REWARD_BASE_NO_PENALTIES not in sample_batch:
        sample_batch[Postprocessing_Custom.REWARD_BASE_NO_PENALTIES] = sample_batch[SampleBatch.REWARDS]

    np_penalty_matrix, np_unweighted_penalty_violation_score, amount_constraint_violations = calculate_penalty_score_risk_autoregressive_ppo(
        sample_batch=sample_batch,
        policy=policy)

    sample_batch[Postprocessing_Custom.UNWEIGHTED_PENALTY_VIOLATION_SCORE] = np_unweighted_penalty_violation_score
    sample_batch[Postprocessing_Custom.AMOUNT_CONSTRAINT_VIOLATIONS] = amount_constraint_violations
    sample_batch[Postprocessing_Custom.BOOL_ANY_CONSTRAINT_VIOLATIONS] = (amount_constraint_violations > 0).astype(int)

    return sample_batch

def calculate_penalty_score_risk_autoregressive_ppo(policy, sample_batch):

    env_config = policy.config.get("env_config")
    conditional_minkowski_encoding_type = policy.config.get("env_config").get("constraints_conditional_minkowski_encoding_type", None)

    if conditional_minkowski_encoding_type is not None:
        head_factor_list = policy.config.get("env_config").get("head_factor_list")
        action_mask_dict = policy.config.get("env_config").get("action_mask_dict")
        list_agg_constraints = generate_aggregated_constraints_conditional_minkowski_encoding(
            head_factor_list,
            action_mask_dict,
            full_constraint_check=True,
            conditional_minkowski_encoding_type=conditional_minkowski_encoding_type)
    else:
        #standard procedure
        list_raw_constraint_tuples = generate_list_raw_constraint_tuples(policy.config.get("env_config"))

        list_relationship_enriched_tuple = convert_raw_constraints_to_full_constraint_tuple(
            list_constraint_tuple=list_raw_constraint_tuples)
        list_agg_constraints = generate_aggregated_constraints(list_relationship_enriched_tuple)

    #2) Generate correct processed action:
    action_processed = calculate_action_allocation(sample_batch[SampleBatch.ACTIONS], env_config, policy.model)

    if isinstance(sample_batch[SampleBatch.ACTIONS], torch.Tensor):
        is_input_tensor = True
    else:
        is_input_tensor = False

    if is_input_tensor:
        np_penalty_matrix = generate_np_penalty_matrix(action_processed,
                                                       list_agg_constraints, only_allow_positive_values=True)
        pass # TODO BE IMPLEMENTED
        raise NotImplementedError

    else:
        np_penalty_matrix = generate_np_penalty_matrix(action_processed,
                                                       list_agg_constraints, only_allow_positive_values=True)

    amount_constraint_violations = check_constraint_violations(np_penalty_matrix,
                                                               sample_batch=sample_batch,
                                                               action_processed=action_processed)

    unweighted_penalty_violation_score = np.sum(np.maximum(np.zeros_like(np_penalty_matrix), np_penalty_matrix), axis=1)

    # HERE WE OVERWRITE THE REWARD # No adjustment needed in this case
    sample_batch[SampleBatch.REWARDS] = sample_batch[SampleBatch.REWARDS]

    return np_penalty_matrix, unweighted_penalty_violation_score, amount_constraint_violations

def calculate_penalty_score_P3O(policy, sample_batch):
    conditional_minkowski_encoding_type = policy.config.get("env_config").get(
        "constraints_conditional_minkowski_encoding_type", None)

    if conditional_minkowski_encoding_type is not None:
        head_factor_list = policy.config.get("env_config").get("head_factor_list")
        action_mask_dict = policy.config.get("env_config").get("action_mask_dict")
        list_agg_constraints = generate_aggregated_constraints_conditional_minkowski_encoding(
            head_factor_list,
            action_mask_dict,
            conditional_minkowski_encoding_type=conditional_minkowski_encoding_type)
    else:
        # standard procedure
        list_raw_constraint_tuples = generate_list_raw_constraint_tuples(policy.config.get("env_config"))

        list_relationship_enriched_tuple = convert_raw_constraints_to_full_constraint_tuple(
            list_constraint_tuple=list_raw_constraint_tuples)
        list_agg_constraints = generate_aggregated_constraints(list_relationship_enriched_tuple)

    env_config = policy.config.get("env_config")

    # Merging only here necessary, since this is the only approach that is capable of applying the multi layer model
    action_processed = calculate_action_allocation(sample_batch[SampleBatch.ACTIONS], env_config)

    if isinstance(sample_batch[SampleBatch.ACTIONS], torch.Tensor):
        is_input_tensor = True
    else:
        is_input_tensor = False

    if is_input_tensor:
        np_penalty_matrix = generate_np_penalty_matrix(action_processed,
                                                       list_agg_constraints, only_allow_positive_values=True)
        pass # TODO BE IMPLEMENTED
        raise NotImplementedError

    else:
        np_penalty_matrix = generate_np_penalty_matrix(action_processed,
                                                       list_agg_constraints, only_allow_positive_values=True)

    amount_constraint_violations = check_constraint_violations(np_penalty_matrix,
                                                               sample_batch=sample_batch,
                                                               action_processed=action_processed)

    unweighted_penalty_violation_score = np.sum(np.maximum(np.zeros_like(np_penalty_matrix), np_penalty_matrix), axis=1)

    return np_penalty_matrix, unweighted_penalty_violation_score, amount_constraint_violations

def compute_cost_values_P3O(policy, sample_batch, other_agent_batches, episode):
    """
    Equivalent to compute_penalty_values
    :param policy:
    :param sample_batch:
    :param other_agent_batches:
    :param episode:
    :return:
    """

    if Postprocessing_Custom.REWARD_BASE_NO_PENALTIES not in sample_batch:
        sample_batch[Postprocessing_Custom.REWARD_BASE_NO_PENALTIES] = sample_batch[SampleBatch.REWARDS]

    np_penalty_matrix, np_unweighted_penalty_violation_score, amount_constraint_violations = calculate_penalty_score_P3O(
        sample_batch=sample_batch,
        policy=policy)

    #sample_batch[Postprocessing_Custom.LOG_BARRIER_CONSTRAINT_PENALTIES] = np_weighted_processed_penalty_value
    sample_batch[Postprocessing_Custom.UNWEIGHTED_PENALTY_VIOLATION_SCORE] = np_unweighted_penalty_violation_score
    sample_batch[Postprocessing_Custom.AMOUNT_CONSTRAINT_VIOLATIONS] = amount_constraint_violations
    sample_batch[Postprocessing_Custom.BOOL_ANY_CONSTRAINT_VIOLATIONS] = (amount_constraint_violations>0).astype(int)

    for idx, column in enumerate(np.transpose(np_penalty_matrix).tolist()):
        sample_batch[f'{SampleBatch_Custom.COST_OBS}{idx}'] = np.array(column)

    return sample_batch

def compute_cost_values_IPO(policy, sample_batch, other_agent_batches, episode):

    if Postprocessing_Custom.REWARD_BASE_NO_PENALTIES not in sample_batch:
        sample_batch[Postprocessing_Custom.REWARD_BASE_NO_PENALTIES] = sample_batch[SampleBatch.REWARDS]

    #TODO
    np_penalty_matrix, np_unweighted_penalty_violation_score, amount_constraint_violations = calculate_penalty_score_ipo(
        sample_batch=sample_batch,
        policy=policy)

    #sample_batch[Postprocessing_Custom.LOG_BARRIER_CONSTRAINT_PENALTIES] = np_weighted_processed_penalty_value
    sample_batch[Postprocessing_Custom.UNWEIGHTED_PENALTY_VIOLATION_SCORE] = np_unweighted_penalty_violation_score
    sample_batch[Postprocessing_Custom.AMOUNT_CONSTRAINT_VIOLATIONS] = amount_constraint_violations
    sample_batch[Postprocessing_Custom.BOOL_ANY_CONSTRAINT_VIOLATIONS] = (amount_constraint_violations>0).astype(int)

    for idx, column in enumerate(np.transpose(np_penalty_matrix).tolist()):
        sample_batch[f'{SampleBatch_Custom.COST_OBS}{idx}'] = np.array(column)

    return sample_batch

def calculate_penalty_score_ipo(policy, sample_batch):
    conditional_minkowski_encoding_type = policy.config.get("env_config").get(
        "constraints_conditional_minkowski_encoding_type", None)

    if conditional_minkowski_encoding_type is not None:
        head_factor_list = policy.config.get("env_config").get("head_factor_list")
        action_mask_dict = policy.config.get("env_config").get("action_mask_dict")
        list_agg_constraints = generate_aggregated_constraints_conditional_minkowski_encoding(
            head_factor_list,
            action_mask_dict,
            conditional_minkowski_encoding_type=conditional_minkowski_encoding_type)
    else:
        # standard procedure
        list_raw_constraint_tuples = generate_list_raw_constraint_tuples(policy.config.get("env_config"))

        list_relationship_enriched_tuple = convert_raw_constraints_to_full_constraint_tuple(
            list_constraint_tuple=list_raw_constraint_tuples)
        list_agg_constraints = generate_aggregated_constraints(list_relationship_enriched_tuple)

    env_config = policy.config.get("env_config")

    # Merging only here necessary, since this is the only approach that is capable of applying the multi layer model
    action_processed = calculate_action_allocation(sample_batch[SampleBatch.ACTIONS], env_config)

    if isinstance(action_processed, torch.Tensor):
        is_input_tensor = True
    else:
        is_input_tensor = False

    if is_input_tensor:
        np_penalty_matrix = generate_np_penalty_matrix(action_processed,
                                                       list_agg_constraints, only_allow_positive_values=True)
        pass # TODO BE IMPLEMENTED
        raise NotImplementedError

    else:
        np_penalty_matrix = generate_np_penalty_matrix(action_processed,
                                                       list_agg_constraints, only_allow_positive_values=True)

    amount_constraint_violations = check_constraint_violations(np_penalty_matrix,
                                                               sample_batch=sample_batch,
                                                               action_processed=action_processed)

    unweighted_penalty_violation_score = np.sum(np.maximum(np.zeros_like(np_penalty_matrix), np_penalty_matrix), axis=1)

    return np_penalty_matrix, unweighted_penalty_violation_score, amount_constraint_violations


def calculate_penalty_score_ipo_merged(policy, sample_batch):

    conditional_minkowski_encoding_type = policy.config.get("env_config").get(
        "constraints_conditional_minkowski_encoding_type", None)

    if conditional_minkowski_encoding_type is not None:
        head_factor_list = policy.config.get("env_config").get("head_factor_list")
        action_mask_dict = policy.config.get("env_config").get("action_mask_dict")
        list_agg_constraints = generate_aggregated_constraints_conditional_minkowski_encoding(
            head_factor_list,
            action_mask_dict,
            conditional_minkowski_encoding_type=conditional_minkowski_encoding_type)
    else:
        # standard procedure
        list_raw_constraint_tuples = generate_list_raw_constraint_tuples(policy.config.get("env_config"))

        list_relationship_enriched_tuple = convert_raw_constraints_to_full_constraint_tuple(
            list_constraint_tuple=list_raw_constraint_tuples)
        list_agg_constraints = generate_aggregated_constraints(list_relationship_enriched_tuple)

    env_config = policy.config.get("env_config")

    # Merging only here necessary, since this is the only approach that is capable of applying the multi layer model
    action_processed = calculate_action_allocation(sample_batch[SampleBatch.ACTIONS], env_config)

    if isinstance(action_processed, torch.Tensor):
        is_input_tensor = True
    else:
        is_input_tensor = False

    if is_input_tensor:
        np_penalty_matrix = generate_np_penalty_matrix(action_processed,
                                                       list_agg_constraints, only_allow_positive_values=True)
        pass # TODO BE IMPLEMENTED
        raise NotImplementedError

    else:
        np_penalty_matrix = generate_np_penalty_matrix(action_processed,
                                                       list_agg_constraints, only_allow_positive_values=True)

    amount_constraint_violations = check_constraint_violations(np_penalty_matrix,
                                                               sample_batch=sample_batch,
                                                               action_processed=action_processed)

    unweighted_penalty_violation_score = np.sum(np.maximum(np.zeros_like(np_penalty_matrix), np_penalty_matrix), axis=1)

    #calculate the log of the penalty
    np_log_penalty_values = np.log(np_penalty_matrix)
    min_penalty_value = -1e+9 #necessary to avoid
    np_processed_penalty_values = np.nan_to_num(np_log_penalty_values, nan=min_penalty_value, neginf=min_penalty_value)
    #print(np_processed_penalty_values)
    t_weight = policy.config.get("t_weight")
    #print(t_weight)
    np_weighted_processed_penalty_value = 1/t_weight*np.sum(np_processed_penalty_values, axis=1) #summing up all penalties

    return np_weighted_processed_penalty_value, unweighted_penalty_violation_score, amount_constraint_violations


def compute_constraint_penalized_rewards_ipo_merged(policy, sample_batch, other_agent_batches, episode):
    """
    NOT USED ANY MORE
    :param policy:
    :param sample_batch:
    :param other_agent_batches:
    :param episode:
    :return:
    """
    if Postprocessing_Custom.REWARD_BASE_NO_PENALTIES not in sample_batch:
        sample_batch[Postprocessing_Custom.REWARD_BASE_NO_PENALTIES] = sample_batch[SampleBatch.REWARDS]

    np_weighted_processed_penalty_value, np_unweighted_penalty_violation_score, amount_constraint_violations\
        = calculate_penalty_score_ipo_merged(
        sample_batch=sample_batch,
        policy=policy)

    sample_batch[Postprocessing_Custom.LOG_BARRIER_CONSTRAINT_PENALTIES] = np_weighted_processed_penalty_value
    sample_batch[Postprocessing_Custom.UNWEIGHTED_PENALTY_VIOLATION_SCORE] = np_unweighted_penalty_violation_score
    sample_batch[Postprocessing_Custom.AMOUNT_CONSTRAINT_VIOLATIONS] = amount_constraint_violations
    sample_batch[Postprocessing_Custom.BOOL_ANY_CONSTRAINT_VIOLATIONS] = (amount_constraint_violations > 0).astype(int)

    # Overwriting reward
    sample_batch[SampleBatch.REWARDS] = sample_batch[SampleBatch.REWARDS] - sample_batch[
        Postprocessing_Custom.LOG_BARRIER_CONSTRAINT_PENALTIES]

    return sample_batch


def calculate_penalty_score_lagrange(policy, sample_batch, np_return=True):

    lambda_model = policy.model.lambda_penalty_model

    conditional_minkowski_encoding_type = policy.config.get("env_config").get(
        "constraints_conditional_minkowski_encoding_type", None)

    if conditional_minkowski_encoding_type is not None:
        head_factor_list = policy.config.get("env_config").get("head_factor_list")
        action_mask_dict = policy.config.get("env_config").get("action_mask_dict")
        list_agg_constraints = generate_aggregated_constraints_conditional_minkowski_encoding(
            head_factor_list,
            action_mask_dict,
            conditional_minkowski_encoding_type=conditional_minkowski_encoding_type)
    else:
        # standard procedure
        list_raw_constraint_tuples = generate_list_raw_constraint_tuples(policy.config.get("env_config"))

        list_relationship_enriched_tuple = convert_raw_constraints_to_full_constraint_tuple(
            list_constraint_tuple=list_raw_constraint_tuples)
        list_agg_constraints = generate_aggregated_constraints(list_relationship_enriched_tuple)

    env_config = policy.config.get("env_config")

    # Merging only here necessary, since this is the only approach that is capable of applying the multi layer model
    action_processed = calculate_action_allocation(sample_batch[SampleBatch.ACTIONS], env_config)

    if isinstance(action_processed, torch.Tensor):
        is_input_tensor = True
    else:
        is_input_tensor = False

    if is_input_tensor:
        np_penalty_matrix = generate_np_penalty_matrix(action_processed,
                                                       list_agg_constraints)
        pass # TODO BE IMPLEMENTED
        raise NotImplementedError

    else:
        np_penalty_matrix = generate_np_penalty_matrix(action_processed,
                                                       list_agg_constraints)
        torch_penalty_model_input = torch.from_numpy(np_penalty_matrix).float().to(lambda_model.availabe_device)


    torch_penalty_scores = lambda_model(torch_penalty_model_input)

    amount_constraint_violations = check_constraint_violations(np_penalty_matrix,
                                                               sample_batch=sample_batch,
                                                               action_processed=action_processed)

    if np_return:
        lambda_weighted_penalty_score = torch_penalty_scores.cpu().detach().numpy().flatten()
        # for analysis purposes we also want the strength (aka sum) of the penalty violations, i.e. only values >= 0
        unweighted_penalty_violation_score = np.sum(np.maximum(np.zeros_like(np_penalty_matrix), np_penalty_matrix), axis=1)
        return lambda_weighted_penalty_score, unweighted_penalty_violation_score, amount_constraint_violations
    else:
        pass


def compute_constraint_penalized_rewards(policy, sample_batch, other_agent_batches, episode):

    if Postprocessing_Custom.REWARD_BASE_NO_PENALTIES not in sample_batch:
        sample_batch[Postprocessing_Custom.REWARD_BASE_NO_PENALTIES] = sample_batch[SampleBatch.REWARDS]

    np_lambda_weighted_penalty_score, np_unweighted_penalty_violation_score, amount_constraint_violations = \
        calculate_penalty_score_lagrange(sample_batch=sample_batch,
                                               policy=policy)
    sample_batch[Postprocessing_Custom.LAMBDA_WEIGHTED_CONSTRAINT_PENALTIES] = np_lambda_weighted_penalty_score
    sample_batch[Postprocessing_Custom.UNWEIGHTED_PENALTY_VIOLATION_SCORE] = np_unweighted_penalty_violation_score
    sample_batch[Postprocessing_Custom.AMOUNT_CONSTRAINT_VIOLATIONS] = amount_constraint_violations
    sample_batch[Postprocessing_Custom.BOOL_ANY_CONSTRAINT_VIOLATIONS] = (amount_constraint_violations>0).astype(int)

    # Overwriting reward #FIXME TODO
    sample_batch[SampleBatch.REWARDS] = sample_batch[SampleBatch.REWARDS] - sample_batch[Postprocessing_Custom.LAMBDA_WEIGHTED_CONSTRAINT_PENALTIES]

    return sample_batch


def compute_cost_advantages(
    rollout: SampleBatch,
    last_cost_val: float,
    index_cost_constraint: int,
    gamma: float = 0.9,
    lambda_: float = 1.0,
    cost_vf_use_gae: bool = True,
    cost_vf_use_critic: bool = True,
):
    """Given a rollout, compute its value targets and the advantages.

    Args:
        rollout: SampleBatch of a single trajectory.
        last_r: Value estimation for last observation.
        gamma: Discount factor.
        lambda_: Parameter for GAE.
        use_gae: Using Generalized Advantage Estimation.
        use_critic: Whether to use critic (value estimates). Setting
            this to False will use 0 as baseline.

    Returns:
        SampleBatch with experience from rollout and processed rewards.
    """
    #f'{SampleBatch_Custom.COST_VF_PREDS}{index_cost_constraint}' <- SampleBatch.VF_PREDS
    #last_cost_val <- last_r
    #f'{SampleBatch_Custom.COST_OBS}{index_cost_constraint}' <- SampleBatch.REWARDS
    #f'{Postprocessing_Custom.COST_ADVANTAGES}{index_cost_constraint}' <- Postprocessing.ADVANTAGES
    #f'{Postprocessing_Custom.COST_VALUE_TARGETS}{index_cost_constraint}' <- Postprocessing.VALUE_TARGETS
    #cost_vf_use_critic <- use_critic
    #cost_vf_use_gae <- use_gae
    assert (
            f'{SampleBatch_Custom.COST_VF_PREDS}{index_cost_constraint}' in rollout or not cost_vf_use_critic
    ), "cost_vf_use_critic=True but values not found"
    assert cost_vf_use_critic or not cost_vf_use_gae, "Can't use gae without using a value function"

    if cost_vf_use_gae:
        vpred_t = np.concatenate(
            [rollout[f'{SampleBatch_Custom.COST_VF_PREDS}{index_cost_constraint}'], np.array([last_cost_val])])
        delta_t = rollout[f'{SampleBatch_Custom.COST_OBS}{index_cost_constraint}'] + gamma * vpred_t[1:] - vpred_t[:-1]
        # This formula for the advantage comes from:
        # "Generalized Advantage Estimation": https://arxiv.org/abs/1506.02438
        rollout[f'{Postprocessing_Custom.COST_ADVANTAGES}{index_cost_constraint}'] = discount_cumsum(delta_t,
                                                                                                     gamma * lambda_)
        rollout[f'{Postprocessing_Custom.COST_VALUE_TARGETS}{index_cost_constraint}'] = (
                rollout[f'{Postprocessing_Custom.COST_ADVANTAGES}{index_cost_constraint}'] + rollout[
            f'{SampleBatch_Custom.COST_VF_PREDS}{index_cost_constraint}']
        ).astype(np.float32)
    else:
        rewards_plus_v = np.concatenate(
            [rollout[f'{SampleBatch_Custom.COST_OBS}{index_cost_constraint}'], np.array([last_cost_val])]
        )
        discounted_returns = discount_cumsum(rewards_plus_v, gamma)[:-1].astype(
            np.float32
        )

        if cost_vf_use_critic:
            rollout[f'{Postprocessing_Custom.COST_ADVANTAGES}{index_cost_constraint}'] = (
                    discounted_returns - rollout[f'{SampleBatch_Custom.COST_VF_PREDS}{index_cost_constraint}']
            )
            rollout[f'{Postprocessing_Custom.COST_VALUE_TARGETS}{index_cost_constraint}'] = discounted_returns
        else:
            rollout[f'{Postprocessing_Custom.COST_ADVANTAGES}{index_cost_constraint}'] = discounted_returns
            rollout[f'{Postprocessing_Custom.COST_VALUE_TARGETS}{index_cost_constraint}'] = np.zeros_like(
                rollout[f'{Postprocessing_Custom.COST_ADVANTAGES}{index_cost_constraint}']
            )

    rollout[f'{Postprocessing_Custom.COST_ADVANTAGES}{index_cost_constraint}'] = rollout[
        f'{Postprocessing_Custom.COST_ADVANTAGES}{index_cost_constraint}'].astype(
        np.float32
    )

    return rollout

def compute_cost_gae_for_sample_batch(
    policy: Policy,
    sample_batch: SampleBatch,
    other_agent_batches: Optional[Dict[AgentID, SampleBatch]] = None,
    episode: Optional[Episode] = None,
) -> SampleBatch:
    """Adds GAE (generalized advantage estimations) to a trajectory for the COSTS

    The trajectory contains only data from one episode and from one agent.
    - If  `config.batch_mode=truncate_episodes` (default), sample_batch may
    contain a truncated (at-the-end) episode, in case the
    `config.rollout_fragment_length` was reached by the sampler.
    - If `config.batch_mode=complete_episodes`, sample_batch will contain
    exactly one episode (no matter how long).
    New columns can be added to sample_batch and existing ones may be altered.

    Args:
        policy: The Policy used to generate the trajectory (`sample_batch`)
        sample_batch: The SampleBatch to postprocess.
        other_agent_batches: Optional dict of AgentIDs mapping to other
            agents' trajectory data (from the same episode).
            NOTE: The other agents use the same policy.
        episode: Optional multi-agent episode object in which the agents
            operated.

    Returns:
        The postprocessed, modified SampleBatch (or a new one).
    """
    for idx_cost in range(policy.model.amount_helper_constraints):
        # Trajectory is actually complete -> last r=0.0.
        if sample_batch[SampleBatch.DONES][-1]:
            last_cost_val = 0.0
        # Trajectory has been truncated -> last r=VF estimate of last obs.
        else:
            # Input dict is provided to us automatically via the Model's
            # requirements. It's a single-timestep (last one in trajectory)
            # input_dict.
            # Create an input dict according to the Model's requirements.
            input_dict = sample_batch.get_single_step_input_dict(
                policy.model.view_requirements, index="last"
            )
            last_cost_val = policy._cost_value(idx_cost=idx_cost, **input_dict)

        # Adds the policy logits, VF preds, and advantages to the batch,
        # using GAE ("generalized advantage estimation") or not.
        batch = compute_cost_advantages(
            rollout=sample_batch,
            last_cost_val=last_cost_val,
            index_cost_constraint=idx_cost,
            gamma=policy.config["gamma"],
            lambda_=policy.config["lambda"],
            cost_vf_use_gae=policy.config["model"]["custom_model_config"]["cost_vf_use_gae"],
            cost_vf_use_critic=policy.config["model"]["custom_model_config"].get("cost_vf_use_critic", True),
        )

    return batch

### RISK COMPONENT ####

def add_processed_action(policy: Policy,
                           sample_batch: SampleBatch,
                           other_agent_batches=None,
                           episode=None):
    env_config = policy.config.get("env_config")

    sample_batch[Postprocessing_Custom.PROCESSED_ACTIONS] = calculate_action_allocation(sample_batch[SampleBatch.ACTIONS], env_config)
    return sample_batch

def add_dummy_sequence_length(policy: Policy,
                           sample_batch: SampleBatch,
                           other_agent_batches=None,
                           episode=None) -> SampleBatch:

    np_dummy = np.ones_like(sample_batch[SampleBatch.REWARDS], dtype=np.int32)
    sample_batch[Postprocessing_Custom.SEQ_LENS_DUMMY] = np.cumsum(np_dummy)
    return sample_batch

def calculate_risk_logic(policy: Policy,
                    sample_batch: SampleBatch,
                    other_agent_batches=None,
                    episode=None):

    sample_batch = add_processed_action(policy, sample_batch, other_agent_batches, episode)
    sample_batch = add_dummy_sequence_length(policy, sample_batch, other_agent_batches, episode)
    sample_batch = add_single_asset_reward_and_penalty(policy, sample_batch, other_agent_batches, episode)

    env_config = policy.config.get("env_config")
    #Calculate risk #else we do single trajectory risk penalty
    if env_config.get("risk_mode")=="risk_per_time_step":
        is_moment_model_attention = policy.model.use_moment_attention
        #Estimating risk
        if is_moment_model_attention:
            #sample_batch, list_moment_state_transformers, is_dummy_run = add_moment_model_state(policy, sample_batch,
            #                                                                  other_agent_batches, episode)
            #sample_batch = estimate_portfolio_risk_attention(policy, sample_batch, other_agent_batches, episode, list_moment_state_transformers, is_dummy_run)
            sample_batch = estimate_portfolio_risk_transformer(policy, sample_batch, other_agent_batches, episode)
        else:
            #single portfolio risk estimate
            #
            sample_batch = estimate_portfolio_risk(policy, sample_batch, other_agent_batches)
    elif env_config.get("risk_mode")=="risk_per_single_trajectory":
        sample_batch = estimate_portfolio_risk_per_single_trajectory(policy, sample_batch, other_agent_batches, episode)
    elif env_config.get("risk_mode")=="risk_MVPI_trajectory":
        sample_batch = estimate_portfolio_risk_MVPI_trajectory(policy, sample_batch, other_agent_batches, episode)
    else:
        raise ValueError("Unknown Risk Mode")

    # HERE WE MUTATE THE SAMPLE.REWARD for later GAE calculation
    sample_batch = compute_rewards_incl_risk_penalty(policy, sample_batch, other_agent_batches, episode)

    return sample_batch


def add_single_asset_reward_and_penalty(policy: Policy,
                           sample_batch: SampleBatch,
                           other_agent_batches=None,
                           episode=None):

    np_input = sample_batch[SampleBatch.OBS]
    environment_class = extract_environment_class_from_config(policy.config)

    dict_decomposed = environment_class.decompose_observation_wrapper_dict(np_input,
                                                                           config=policy.config.get("env_config"))

    prev_returns = dict_decomposed["prev_observed_returns"]

    sliced_prev_return = prev_returns[1:,:]

    #The last episode needs to be imputed, since we do not have this information passed from the environment
    #Estimate is average performance in this episode weighted by the final action and then normalized by the actual observed total reward

    #Fix 02.01.2023
    imputed_single_reward = np.expand_dims(np.mean(prev_returns, axis=0), axis=0)

    #depreciated
    #risk_invested_percentage = sample_batch[SampleBatch.ACTIONS][-1,:][~np.isclose(np.mean(prev_returns, axis=0), 0)].sum()
    #value_to_impute = sample_batch[SampleBatch.REWARDS][-1]/risk_invested_percentage
    #imputed_single_reward = np.ones_like(np.mean(prev_returns, axis=0))*value_to_impute

    #imputed_single_reward = np.expand_dims(np.where(np.isclose(np.mean(prev_returns, axis=0), 0), 0, imputed_single_reward), 0)
    imputed_prev_returns = np.concatenate((sliced_prev_return, imputed_single_reward), 0)#Here we insert the last observation of rewards (which we had to impute)

    if policy.config.get("env_config").get("include_risk_penalty_in_state"):
        sample_batch[Postprocessing_Custom.RISK_PENALTY_PARAMETER_VALUE] = dict_decomposed["risk_penalty"]
    else:
        sample_batch[Postprocessing_Custom.RISK_PENALTY_PARAMETER_VALUE] = \
            np.ones_like(sample_batch[SampleBatch.REWARDS])*policy.config.get("env_config").get("risk_penalty_factor")

    sample_batch[Postprocessing_Custom.REWARDS_DETAILS] = imputed_prev_returns
    return sample_batch



def estimate_portfolio_risk_transformer(policy: Policy,
                           sample_batch: SampleBatch,
                           other_agent_batches=None,
                           episode=None):


    moment_model = policy.model.moment_submodel
    amount_assets = moment_model.number_input_assets

    dataset = generate_trainable_dataset(moment_submodel=moment_model, train_batch=sample_batch)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    for i, (x_batch, y_batch) in enumerate(data_loader):

        y_pred = moment_model(x_batch, y_batch[:, :-1])
        y_pred_first, y_pred_second = moment_model.split_results(y_pred)

    torch_econ_first_moment_return_info, torch_econ_second_moment_return_info = construct_econ_returns(moment_model,
                                                                                                       sample_batch)
    # Calculate MSE: #we calcuate for each element, this is why we have to sum
    first_moment_mse = torch.sum(torch.square((y_pred_first - torch_econ_first_moment_return_info)), dim=1)
    second_moment_mse = torch.sum(torch.square((y_pred_second - torch_econ_second_moment_return_info)), dim=1)

    np_first_moment_mse = first_moment_mse.cpu().detach().numpy()
    np_second_moment_mse = second_moment_mse.cpu().detach().numpy()

    np_merged_pred = torch.squeeze(y_pred, 1).cpu().detach().numpy()

    # FIXME This logic is only valid if we have a non-portfolio return view
    np_flattened_covariance_matrix = \
        reconstruct_flattened_covariance_matrix_from_first_moment_and_second_moment_information(
            flattened_first_moment_and_second_moment_information=np_merged_pred,
            n_assets=amount_assets)

    #FIXME ONLY TO TRACK THE QUALITY of

    np_tmp_info = np.concatenate([np_merged_pred[:,:amount_assets], np_flattened_covariance_matrix], 1)
    sample_batch[Postprocessing_Custom.DEBUG_MOMENTS] = np_tmp_info

    #np_actions = sample_batch[SampleBatch.ACTIONS]
    np_actions = sample_batch[Postprocessing_Custom.PROCESSED_ACTIONS]

    np_action_flattened_covariance_matrix = np.concatenate((np_actions, np_flattened_covariance_matrix), axis=1)

    np_estimated_portfolio_variance = calculate_estimated_portfolio_variance_from_action(
        np_action_flattened_covariance_matrix=np_action_flattened_covariance_matrix,
        n_assets=amount_assets)

    # to ensure te right dimensionality consistent with the SAMPLE:REWARD
    np_estimated_portfolio_variance = np_estimated_portfolio_variance.squeeze()

    sample_batch[Postprocessing_Custom.RISK_VARIANCE_ESTIMATE] = np_estimated_portfolio_variance
    sample_batch[Postprocessing_Custom.MSE_FIRST_MOMENT] = np_first_moment_mse
    sample_batch[Postprocessing_Custom.MSE_SECOND_MOMENT] = np_second_moment_mse

    return sample_batch


def compute_rewards_incl_risk_penalty(policy: Policy,
                           sample_batch: SampleBatch,
                           other_agent_batches=None,
                           episode=None) -> SampleBatch:

    """
    Overwrites the plain REWARD of the environment and replaces it with a REWARD adjusted by the risk penalty
    The plain reward is renamed
    :param rollout:
    :param risk_penalty_factor:
    :return:
    """

    # Store base rewards, before overwriting the initial rewards
    sample_batch[Postprocessing_Custom.REWARDS_BASE_WITHOUT_RISK_PENALTY] = sample_batch[SampleBatch.REWARDS]

    risk_mode = policy.config.get("env_config").get("risk_mode")

    if risk_mode == "risk_per_time_step":

        np_standard_deviation_estimate = np.sqrt(sample_batch[Postprocessing_Custom.RISK_VARIANCE_ESTIMATE])
        sample_batch[Postprocessing_Custom.RISK_PENALTY] = sample_batch[Postprocessing_Custom.RISK_PENALTY_PARAMETER_VALUE] * np_standard_deviation_estimate
        sample_batch[Postprocessing_Custom.REWARDS_INCL_RISK] = sample_batch[SampleBatch.REWARDS] - sample_batch[Postprocessing_Custom.RISK_PENALTY]

        # HERE WE OVERWRITE THE REWARD (which is then later used to calculate the GAE)
        sample_batch[SampleBatch.REWARDS] = sample_batch[SampleBatch.REWARDS] - sample_batch[Postprocessing_Custom.RISK_PENALTY_PARAMETER_VALUE] * np_standard_deviation_estimate

        return sample_batch

    elif risk_mode=="risk_per_single_trajectory":
        portfolio_trajectory_variance = np.sum(sample_batch[Postprocessing_Custom.RISK_VARIANCE_ESTIMATE])

        tmp_variance = np.zeros_like(sample_batch[SampleBatch.REWARDS])
        tmp_variance[-1] = portfolio_trajectory_variance
        np_standard_deviation_trajectory_estimate = np.sqrt(tmp_variance)

        sample_batch[Postprocessing_Custom.RISK_PENALTY] = sample_batch[Postprocessing_Custom.RISK_PENALTY_PARAMETER_VALUE] * np_standard_deviation_trajectory_estimate
        sample_batch[Postprocessing_Custom.REWARDS_INCL_RISK] = sample_batch[SampleBatch.REWARDS] - sample_batch[Postprocessing_Custom.RISK_PENALTY]
        # HERE WE OVERWRITE THE REWARD (which is then later used to calculate the GAE)
        sample_batch[SampleBatch.REWARDS] = sample_batch[SampleBatch.REWARDS] - sample_batch[Postprocessing_Custom.RISK_PENALTY]

        return sample_batch

    elif risk_mode == "risk_MVPI_trajectory":

        if Postprocessing_Custom.RISK_MVPI_FORMULATION in sample_batch:
            sample_batch[SampleBatch.REWARDS] = sample_batch[SampleBatch.REWARDS] - \
                                                sample_batch[Postprocessing_Custom.RISK_PENALTY_PARAMETER_VALUE] * sample_batch[Postprocessing_Custom.RISK_MVPI_FORMULATION]
        else:
            pass #do not mutate the reward
        return sample_batch
        #rewards = rewards - config.lam * rewards.pow(2) + 2 * config.lam * rewards * y
        #rewards = rewards - lam(rewards.pow(2)+2rewards*y)
    else:
        raise ValueError("No risk mode given")