import argparse
import os

import ray
import numpy as np
from ray import air, tune
from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes


import collections
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import ray
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils.typing import GradInfoDict, LearnerStatsDict, ResultDict
from ray.rllib.evaluation.metrics import RolloutMetrics

from custom_keys import Postprocessing_Custom
import wandb

from helper_functions import generate_allocation_bar_chart, write_constraint_violations



def custom_summarize_episodes_risk_aspect(
    episodes: List[RolloutMetrics],
    new_episodes: List[RolloutMetrics] = None,
    keep_custom_metrics: bool = False):

    percentile_bound = (0.10, 0.90)

    if new_episodes is None:
        new_episodes = episodes

    episode_rewards = []
    episode_lengths = []
    policy_rewards = collections.defaultdict(list)
    custom_metrics = collections.defaultdict(list)
    perf_stats = collections.defaultdict(list)
    hist_stats = collections.defaultdict(list)
    episode_media = collections.defaultdict(list)
    num_faulty_episodes = 0

    episode_rewards = []
    #episode_rewards_without_risk_penalty = []
    episode_lengths = []
    #episode_rewards_incl_risk_penalty = []
    #episode_risk_penalty = []
    episode_mse_first_moment = []
    episode_mse_second_moment = []

    episode_moment_debug = []

    policy_rewards = collections.defaultdict(list)
    custom_metrics = collections.defaultdict(list)
    perf_stats = collections.defaultdict(list)
    hist_stats = collections.defaultdict(list)
    episode_media = collections.defaultdict(list)

    list_relevant_keys = [Postprocessing_Custom.MSE_FIRST_MOMENT, Postprocessing_Custom.MSE_SECOND_MOMENT,
                          Postprocessing_Custom.RISK_PENALTY, Postprocessing_Custom.REWARDS_INCL_RISK,
                          Postprocessing_Custom.REWARDS_BASE_WITHOUT_RISK_PENALTY]

    episode_dict = dict(zip(list_relevant_keys, [ [] for _ in range(len(list_relevant_keys)) ]))

    for episode in episodes:
        episode_lengths.append(episode.episode_length)
        episode_rewards.append(episode.episode_reward)

        for key in list_relevant_keys:
            if key in episode.custom_metrics:
                if isinstance(episode.custom_metrics.get(key), list):
                    #This only works for lists
                    episode_dict.get(key).append(
                        episode.custom_metrics.get(key)) #np.sum([np.array([3,2]), np.array([3,2])])=10 -> works

    return_dict = {}
    if episode_rewards:
        for key, value in episode_dict.items():
            return_dict[key] = np.mean(value)



    list_rewards_base_without_risk_penalty = episode_dict.get(Postprocessing_Custom.REWARDS_BASE_WITHOUT_RISK_PENALTY)

    np_rewards_base_without_risk_penalty = np.array(list_rewards_base_without_risk_penalty).flatten()

    if episode_rewards:
        return_dict[f'{Postprocessing_Custom.REWARDS_BASE_WITHOUT_RISK_PENALTY}_variance'] =\
            np.var(np_rewards_base_without_risk_penalty)
        return_dict[f'{Postprocessing_Custom.REWARDS_BASE_WITHOUT_RISK_PENALTY}_standard_deviation'] = \
            np.sqrt(np.var(np_rewards_base_without_risk_penalty))
        return_dict[f'{Postprocessing_Custom.REWARDS_BASE_WITHOUT_RISK_PENALTY}_min'] = \
            min(np_rewards_base_without_risk_penalty)
        return_dict[f'{Postprocessing_Custom.REWARDS_BASE_WITHOUT_RISK_PENALTY}_max'] = \
            max(np_rewards_base_without_risk_penalty)
        return_dict[f'{Postprocessing_Custom.REWARDS_BASE_WITHOUT_RISK_PENALTY}_perc_low'] = \
            np.percentile(np_rewards_base_without_risk_penalty, percentile_bound[0] * 100)
        return_dict[f'{Postprocessing_Custom.REWARDS_BASE_WITHOUT_RISK_PENALTY}_perc_high'] = \
            np.percentile(np_rewards_base_without_risk_penalty, percentile_bound[1] * 100)

    return return_dict


def generate_confidence_metrics(episode_dict, key_for_metrics_to_be_evaluated, return_dict, list_tuple_standard_deviations):

    list_key_metric = episode_dict.get(key_for_metrics_to_be_evaluated)  # can contain. e.g. the aggregated reward

    np_key_metric_flattened = np.array(list_key_metric).flatten()
    number_observations = np_key_metric_flattened.size

    key_metric_scalar_mean = np.mean(np_key_metric_flattened)
    key_metric_scalar_std = np.sqrt(np.var(np_key_metric_flattened))

    for standard_deviation_factors in list_tuple_standard_deviations:
        return_dict[f'{key_for_metrics_to_be_evaluated}_CI_low_{str(standard_deviation_factors[0])}'] = \
            key_metric_scalar_mean + np.float32(standard_deviation_factors[0])*(key_metric_scalar_std/np.sqrt(number_observations))
        return_dict[f'{key_for_metrics_to_be_evaluated}_CI_high_{str(standard_deviation_factors[1])}'] = \
            key_metric_scalar_mean + np.float32(standard_deviation_factors[1])*(key_metric_scalar_std/np.sqrt(number_observations))

    return return_dict



def generate_distributional_metrics(episode_dict, key_for_metrics_to_be_evaluated, return_dict, list_tuple_percentile_bound):

    list_key_metric = episode_dict.get(key_for_metrics_to_be_evaluated) # can contain. e.g. the aggregated reward

    np_key_metric_flattened = np.array(list_key_metric).flatten()

    return_dict[f'{key_for_metrics_to_be_evaluated}_variance'] = \
        np.var(np_key_metric_flattened)
    return_dict[f'{key_for_metrics_to_be_evaluated}_standard_deviation'] = \
        np.sqrt(np.var(np_key_metric_flattened))
    return_dict[f'{key_for_metrics_to_be_evaluated}_min'] = \
        min(np_key_metric_flattened)
    return_dict[f'{key_for_metrics_to_be_evaluated}_max'] = \
        max(np_key_metric_flattened)
    for percentile_bound in list_tuple_percentile_bound:
        return_dict[f'{key_for_metrics_to_be_evaluated}_perc_low_{str(percentile_bound[0])}'] = \
            np.percentile(np_key_metric_flattened, percentile_bound[0] * 100)
        return_dict[f'{key_for_metrics_to_be_evaluated}_perc_high_{str(percentile_bound[1])}'] = \
            np.percentile(np_key_metric_flattened, percentile_bound[1] * 100)

    return return_dict

def custom_summarize_episodes_lambda_penalties(
    episodes: List[RolloutMetrics],
    new_episodes: List[RolloutMetrics] = None,
    keep_custom_metrics: bool = False,
    custom_experimental_path = None,
    evaluate_risk_metrics = False,
) -> ResultDict:
    """Summarizes a set of episode metrics tuples.
    Args:
        episodes: List of most recent n episodes. This may include historical ones
            (not newly collected in this iteration) in order to achieve the size of
            the smoothing window.
        new_episodes: All the episodes that were completed in this iteration.
    """

    if new_episodes is None:
        new_episodes = episodes

    episode_rewards = []
    episode_lengths = []
    policy_rewards = collections.defaultdict(list)
    custom_metrics = collections.defaultdict(list)
    perf_stats = collections.defaultdict(list)
    hist_stats = collections.defaultdict(list)
    episode_media = collections.defaultdict(list)
    num_faulty_episodes = 0


    list_relevant_keys_mean = [Postprocessing_Custom.UNWEIGHTED_PENALTY_VIOLATION_SCORE, Postprocessing_Custom.AMOUNT_CONSTRAINT_VIOLATIONS,
                               Postprocessing_Custom.BOOL_ANY_CONSTRAINT_VIOLATIONS]
    list_relevant_keys_sum = [Postprocessing_Custom.REWARD_BASE_NO_PENALTIES]

    list_relevant_keys_media = [Postprocessing_Custom.ALLOCATION_mean, Postprocessing_Custom.ALLOCATION_min,
                                Postprocessing_Custom.ALLOCATION_max]

    list_relevant_keys_violation_media = [Postprocessing_Custom.ACTION_VIOLATIONS, Postprocessing_Custom.ACTION_VIOLATIONS_CONSTRAINT_VIOLATIONS
                                          ]

    # This just states for whats keys we should also calculate the std
    list_relevant_keys_calculate_std = [Postprocessing_Custom.REWARD_BASE_NO_PENALTIES]

    list_all_relevant_keys = list_relevant_keys_mean + list_relevant_keys_sum + list_relevant_keys_media + list_relevant_keys_violation_media

    episode_dict = dict(zip(list_all_relevant_keys, [ [] for _ in range(len(list_all_relevant_keys)) ]))

    for episode in episodes:

        episode_lengths.append(episode.episode_length)
        episode_rewards.append(episode.episode_reward)

        for key in list_all_relevant_keys:
            if key in episode.custom_metrics:
                if isinstance(episode.custom_metrics.get(key), list):
                    #This only works for lists
                    episode_dict.get(key).append(
                        episode.custom_metrics.get(key)) # we append single values
            elif key in episode.media:
                if isinstance(episode.media.get(key), list):
                    #This only works for lists
                    episode_dict.get(key).extend( #we extend by entire lists
                        episode.media.get(key))
    return_dict = {}
    if episode_rewards:

        for relevant_key_mean in list_relevant_keys_mean:
            if relevant_key_mean in episode_dict:
                return_dict[relevant_key_mean] = np.mean(episode_dict.get(relevant_key_mean))
                if relevant_key_mean in list_relevant_keys_calculate_std:
                    return_dict[f'{relevant_key_mean}_std'] = np.std(episode_dict.get(relevant_key_mean))

        for relevant_key_sum in list_relevant_keys_sum:
            if relevant_key_sum in episode_dict:
                return_dict[relevant_key_sum] = np.mean(episode_dict.get(relevant_key_sum))
                # here we average those keys which have been summed over the trajectory
                if relevant_key_sum in list_relevant_keys_calculate_std:
                    return_dict[f'{relevant_key_sum}_std'] = np.std(episode_dict.get(relevant_key_sum))


    #Violation stuff
    if Postprocessing_Custom.ACTION_VIOLATIONS_CONSTRAINT_VIOLATIONS in episode.media and Postprocessing_Custom.ACTION_VIOLATIONS in episode.media:
        #tmp_array_violation_index_constraint = np.array(episode_dict.get(Postprocessing_Custom.ACTION_VIOLATIONS_CONSTRAINT_VIOLATIONS))

        list_action_violation_constrain_violation = episode_dict.get(Postprocessing_Custom.ACTION_VIOLATIONS_CONSTRAINT_VIOLATIONS)
        list_action_violation_constrain_violation = [entry[0] for entry in list_action_violation_constrain_violation]
        tmp_array_violation_index_constraint = np.concatenate(list_action_violation_constrain_violation)

        list_action_violations = episode_dict.get(
            Postprocessing_Custom.ACTION_VIOLATIONS)
        list_action_violations = [entry[0] for entry in list_action_violations]
        tmp_array_violation_action = np.concatenate(list_action_violations)

        write_constraint_violations(path=custom_experimental_path,
                                    np_violating_action=tmp_array_violation_action,
                                    np_violation_constraint_index=tmp_array_violation_index_constraint)

    #Allocation stuff
    if Postprocessing_Custom.ALLOCATION_mean in episode.media:
        np_full_mean = np.mean(np.array(episode_dict.get(Postprocessing_Custom.ALLOCATION_mean)), axis=0)
        plt = generate_allocation_bar_chart(np_full_mean)
        return_dict[f'{Postprocessing_Custom.ALLOCATION_mean}_histo'] = \
            wandb.Image(plt)

    if Postprocessing_Custom.ALLOCATION_max in episode.media:
        np_full_max = np.max(np.array(episode_dict.get(Postprocessing_Custom.ALLOCATION_mean)), axis=0)
        plt = generate_allocation_bar_chart(np_full_max)
        return_dict[f'{Postprocessing_Custom.ALLOCATION_max}_histo'] = \
            wandb.Image(plt)

    if Postprocessing_Custom.ALLOCATION_min in episode.media:
        np_full_min = np.min(np.array(episode_dict.get(Postprocessing_Custom.ALLOCATION_mean)), axis=0)
        plt = generate_allocation_bar_chart(np_full_min)
        return_dict[f'{Postprocessing_Custom.ALLOCATION_min}_histo'] = \
            wandb.Image(plt)

    if evaluate_risk_metrics:
        tmp_dict = custom_summarize_episodes_risk_aspect(episodes)
        return_dict = {**return_dict, **tmp_dict}


    if episode_rewards:
        list_tuple_percentile_bound = [(0.10, 0.90), (0.05, 0.95)]
        return_dict = generate_distributional_metrics(
            episode_dict=episode_dict,
            key_for_metrics_to_be_evaluated=Postprocessing_Custom.REWARD_BASE_NO_PENALTIES,
            return_dict=return_dict,
            list_tuple_percentile_bound=list_tuple_percentile_bound,
        )

    if episode_rewards:
        list_tuple_standard_deviations = [(-1.96,1.96)]
        return_dict = generate_confidence_metrics(
            episode_dict=episode_dict,
            key_for_metrics_to_be_evaluated=Postprocessing_Custom.REWARD_BASE_NO_PENALTIES,
            return_dict=return_dict,
            list_tuple_standard_deviations=list_tuple_standard_deviations
        )

    return return_dict

def run_backtesting_analysis(algorithm, eval_workers):

    # Since backtesting is a purely deterministic evaluation, we only need to run a single trajectory
    # Calling .sample() runs exactly one episode per worker due to how the
    # eval workers are configured.

    experiment_path = algorithm._result_logger.logdir
    #pick one single remote worker

    backtesting_workers = [eval_workers.remote_workers()[0]]

    #FIXME we need one worker / one environment
    ray.get([w.foreach_env.remote(lambda env: env.reset())
             for w in backtesting_workers])

    ray.get([w.foreach_env.remote(lambda env: env.set_backtesting_mode(is_backtesting_mode=True))  # activate backtesting mode
             for w in backtesting_workers])

    #we only sample a single episode here
    ray.get([w.sample.remote() for w in backtesting_workers])

    episodes, _ = collect_episodes(
        remote_workers=backtesting_workers, timeout_seconds=99999
    )

    # You can compute metrics from the episodes manually, or use the
    # convenient `summarize_episodes()` utility:
    tmp_backtesting_metrics = custom_summarize_episodes_lambda_penalties(episodes, custom_experimental_path=experiment_path)

    ray.get([w.foreach_env.remote(lambda env: env.set_backtesting_mode(is_backtesting_mode=False))  # deactivate backtesting mode
             for w in backtesting_workers])

    #renaming the keys
    tmp_backtesting_metrics = {f'{key}_backtesting': value for key,value in tmp_backtesting_metrics.items()}

    return tmp_backtesting_metrics

def custom_eval_function_lambda_penalty(algorithm, eval_workers):
    """Example of a custom evaluation function.
    Args:
        algorithm: Algorithm class to evaluate.
        eval_workers: Evaluation WorkerSet.
    Returns:
        metrics: Evaluation metrics dict.
    """

    #print("BIG SHORT")
    experiment_path = algorithm._result_logger.logdir #can be used to directly log
    #print("BIG SHORT")

    evaluation_amount_episodes = algorithm.get_policy().config.get("evaluation_amount_episodes", 100)

    worker_config = algorithm.get_policy().config
    evaluate_risk_metrics = ("custom_model_config" in worker_config.get("model") and "config_moment_model" in worker_config.get(
                "model").get("custom_model_config"))

    for i in range(evaluation_amount_episodes):
        #print("Custom evaluation round", i)
        # Calling .sample() runs exactly one episode per worker due to how the
        # eval workers are configured.
        ray.get([w.foreach_env.remote(lambda env: env.reset())
                 for w in eval_workers.remote_workers()])

        ray.get([w.sample.remote() for w in eval_workers.remote_workers()])


    # Collect the accumulated episodes on the workers, and then summarize the
    # episode stats into a metrics dict.
    episodes, _ = collect_episodes(
        remote_workers=eval_workers.remote_workers(), timeout_seconds=99999
    )
    # You can compute metrics from the episodes manually, or use the
    # convenient `summarize_episodes()` utility:
    metrics = custom_summarize_episodes_lambda_penalties(episodes, custom_experimental_path=experiment_path)
    # Note that the above two statements are the equivalent of:
    # metrics = collect_metrics(eval_workers.local_worker(),
    #                           eval_workers.remote_workers())

    if algorithm.get_policy().config.get("evaluation_allow_backtesting", False):
        dict_metrics_backtesting = run_backtesting_analysis(algorithm, eval_workers)
        metrics = {**metrics, **dict_metrics_backtesting}

    # You can also put custom values in the metrics dict.
    return metrics