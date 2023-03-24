from typing import Dict, Optional, TYPE_CHECKING
import numpy as np

from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.utils.deprecation import deprecation_warning
from ray.rllib.utils.typing import AgentID, PolicyID
from ray.rllib.evaluation import RolloutWorker

import custom_keys
from custom_keys import Postprocessing_Custom
from environment_wrapper import BasicWrapperFinancialEnvironmentShortSelling
from financial_markets_gym.envs.financial_markets_env import FinancialMarketsEnv
from ray.rllib.algorithms.callbacks import DefaultCallbacks

from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

import numpy as np

from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
#from environment_wrapper import BasicWrapperFinancialEnvironmentShortSelling

from helper_functions import convert_raw_constraints_to_full_constraint_tuple, generate_aggregated_constraints, \
    generate_penalty_vec_from_samples_for_agg_constraint_satisfaction, check_constraint_violations, \
    calculate_action_allocation, filter_out_dummy_constraint_violations

class EvaluationLoggerCallback(DefaultCallbacks):

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        self.algo_handler = None
        super().__init__(*args, **kwargs)

    def on_algorithm_init(
            self,
            *,
            algorithm: "Algorithm",
            **kwargs,
    ) -> None:
        """Callback run when a new algorithm instance has finished setup.

        This method gets called at the end of Algorithm.setup() after all
        the initialization is done, and before actually training starts.

        Args:
            algorithm: Reference to the trainer instance.
            kwargs: Forward compatibility placeholder.
        """
        #create a handler for the algorithm, to access the logger
        self.algo_handler = algorithm
        print("WE INITIATED THE ALGORITHM")
        print("-----")

        self.path_variable = "UPGRADED"
        print(self.algo_handler._result_logger.logdir)

    def on_postprocess_trajectory(
        self,
        *,
        worker: "RolloutWorker",
        episode: Episode,
        agent_id: AgentID,
        policy_id: PolicyID,
        policies: Dict[PolicyID, Policy],
        postprocessed_batch: SampleBatch,
        original_batches: Dict[AgentID, Tuple[Policy, SampleBatch]],
        **kwargs,
    ) -> None:
        """Called immediately after a policy's postprocess_fn is called.
        You can use this callback to do additional postprocessing for a policy,
        including looking at the trajectory data of other agents in multi-agent
        settings.
        Args:
            worker: Reference to the current rollout worker.
            episode: Episode object.
            agent_id: Id of the current agent.
            policy_id: Id of the current policy for the agent.
            policies: Mapping of policy id to policy objects. In single
                agent mode there will only be a single "default_policy".
            postprocessed_batch: The postprocessed sample batch
                for this agent. You can mutate this object to apply your own
                trajectory postprocessing.
            original_batches: Mapping of agents to their unpostprocessed
                trajectory data. You should not mutate this object.
            kwargs: Forward compatibility placeholder.
        """

        #Necessary to have visibility of the postprocessed batch keys also in the custom_metrics
        list_relevant_keys_mean = [Postprocessing_Custom.UNWEIGHTED_PENALTY_VIOLATION_SCORE,
                                   Postprocessing_Custom.AMOUNT_CONSTRAINT_VIOLATIONS,
                                   Postprocessing_Custom.BOOL_ANY_CONSTRAINT_VIOLATIONS]

        list_relevant_keys_sum = [Postprocessing_Custom.REWARD_BASE_NO_PENALTIES]

        list_relevant_keys = list_relevant_keys_sum + list_relevant_keys_mean

        for key in list_relevant_keys:
            if key in postprocessed_batch:
                if f"tmp_{key}" not in episode.custom_metrics:
                    episode.custom_metrics[f"tmp_{key}"] = []
                episode.custom_metrics[f"tmp_{key}"].extend(
                    postprocessed_batch.get(key).tolist())


        if Postprocessing_Custom.ACTION_VIOLATIONS_CONSTRAINT_VIOLATIONS in postprocessed_batch \
                and Postprocessing_Custom.ACTION_VIOLATIONS in postprocessed_batch:

            np_violation_samples_filtered, np_index_violation_mask_filtered = filter_out_dummy_constraint_violations(
                np_violation_samples=postprocessed_batch.get(Postprocessing_Custom.ACTION_VIOLATIONS),
               np_index_violation_mask=postprocessed_batch.get(Postprocessing_Custom.ACTION_VIOLATIONS_CONSTRAINT_VIOLATIONS))

            if np_violation_samples_filtered.size>0 and np_index_violation_mask_filtered.size>0:
                if Postprocessing_Custom.ACTION_VIOLATIONS not in episode.media:
                    episode.media[f"tmp_{Postprocessing_Custom.ACTION_VIOLATIONS}"] = []
                episode.media[f"tmp_{Postprocessing_Custom.ACTION_VIOLATIONS}"].append(np_violation_samples_filtered)
                if Postprocessing_Custom.ACTION_VIOLATIONS_CONSTRAINT_VIOLATIONS not in episode.media:
                    episode.media[f"tmp_{Postprocessing_Custom.ACTION_VIOLATIONS_CONSTRAINT_VIOLATIONS}"] = []
                episode.media[f"tmp_{Postprocessing_Custom.ACTION_VIOLATIONS_CONSTRAINT_VIOLATIONS}"].append(np_index_violation_mask_filtered)



        #Risk relevant
        worker_config = worker.get_policy().config
        if "custom_model_config" in worker_config.get("model") and "config_moment_model" in worker_config.get(
                "model").get("custom_model_config"):
            list_relevant_keys = [Postprocessing_Custom.MSE_FIRST_MOMENT, Postprocessing_Custom.MSE_SECOND_MOMENT,
                                  Postprocessing_Custom.DEBUG_MOMENTS,
                                  Postprocessing_Custom.RISK_VARIANCE_ESTIMATE,
                                  Postprocessing_Custom.RISK_PENALTY, Postprocessing_Custom.REWARDS_INCL_RISK,
                                  Postprocessing_Custom.REWARDS_BASE_WITHOUT_RISK_PENALTY]

            for key in list_relevant_keys:
                if key in postprocessed_batch:
                    if f"tmp_{key}" not in episode.custom_metrics:
                        episode.custom_metrics[f"tmp_{key}"] = []
                    episode.custom_metrics[f"tmp_{key}"].extend(
                        postprocessed_batch.get(key).tolist())

    def on_episode_end(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Union[Episode, EpisodeV2, Exception],
        **kwargs,
    ) -> None:
        """Runs when an episode is done.
        Args:
            worker: Reference to the current rollout worker.
            base_env: BaseEnv running the episode. The underlying
                sub environment objects can be retrieved by calling
                `base_env.get_sub_environments()`.
            policies: Mapping of policy id to policy
                objects. In single agent mode there will only be a single
                "default_policy".
            episode: Episode object which contains episode
                state. You can use the `episode.user_data` dict to store
                temporary data, and `episode.custom_metrics` to store custom
                metrics for the episode.
                In case of environment failures, episode may also be an Exception
                that gets thrown from the environment before the episode finishes.
                Users of this callback may then handle these error cases properly
                with their custom logics.
            kwargs: Forward compatibility placeholder.
        """

        """        
        list_relevant_keys = [Postprocessing_Custom.MSE_FIRST_MOMENT, Postprocessing_Custom.MSE_SECOND_MOMENT,
                              Postprocessing_Custom.RISK_PENALTY, Postprocessing_Custom.REWARDS_INCL_RISK,
                              Postprocessing_Custom.REWARDS_BASE_WITHOUT_RISK_PENALTY
                              ]

        
        for key in list_relevant_keys:
            if f"tmp_{key}" in episode.custom_metrics:
                if key not in episode.custom_metrics:
                    episode.custom_metrics[key] = []
                episode.custom_metrics[key].append(np.sum(episode.custom_metrics[f"tmp_{key}"]))
                # removing key
                episode.custom_metrics.pop(f"tmp_{key}", None)

        list_relevant_keys_no_aggregation = [
            Postprocessing_Custom.DEBUG_MOMENTS,
            Postprocessing_Custom.RISK_VARIANCE_ESTIMATE,
            SampleBatch.ACTIONS
        ]

        for key in list_relevant_keys_no_aggregation:
            if f"tmp_{key}" in episode.custom_metrics:
                if key not in episode.custom_metrics:
                    episode.custom_metrics[key] = []
                episode.custom_metrics[key].append(episode.custom_metrics[f"tmp_{key}"])

                # removing key
                episode.custom_metrics.pop(f"tmp_{key}", None)

        ###process allocation
        env_config = worker.get_policy().config.get("env_config")
        list_raw_constraint_tuples = self.generate_list_raw_constraint_tuples(env_config)
        list_relationship_enriched_tuple = convert_raw_constraints_to_full_constraint_tuple(
            list_constraint_tuple=list_raw_constraint_tuples)
        list_agg_constraints = generate_aggregated_constraints(list_relationship_enriched_tuple)
        #print(list_agg_constraints)

        if f'tmp_{Postprocessing_Custom.ALLOCATION}' in episode.media:
            allocation_full = np.array(episode.media[f'tmp_{Postprocessing_Custom.ALLOCATION}'])
            if Postprocessing_Custom.ALLOCATION_mean not in episode.media:
                episode.media[Postprocessing_Custom.ALLOCATION_mean] = []
            episode.media[Postprocessing_Custom.ALLOCATION_mean].append(np.mean(allocation_full, axis=0))
            if Postprocessing_Custom.ALLOCATION_max not in episode.media:
                episode.media[Postprocessing_Custom.ALLOCATION_max] = []
            episode.media[Postprocessing_Custom.ALLOCATION_max].append(np.max(allocation_full, axis=0))
            if Postprocessing_Custom.ALLOCATION_min not in episode.media:
                episode.media[Postprocessing_Custom.ALLOCATION_min] = []
            episode.media[Postprocessing_Custom.ALLOCATION_min].append(np.min(allocation_full, axis=0))

            list_penalty_vectors = generate_penalty_vec_from_samples_for_agg_constraint_satisfaction(allocation_full,
                                                                                                     list_agg_constraints)

            amount_allocations_in_violation = check_constraint_violations(list_agg_constraints, list_penalty_vectors)
            if Postprocessing_Custom.AMOUNT_ALLOCATION_VIOLATIONS not in episode.custom_metrics:
                episode.custom_metrics[Postprocessing_Custom.AMOUNT_ALLOCATION_VIOLATIONS] = []

            episode.custom_metrics[Postprocessing_Custom.AMOUNT_ALLOCATION_VIOLATIONS].append(amount_allocations_in_violation)

            #print(f'{amount_allocations_in_violation} allocations in violation')

            #print(list_penalty_vectors)
            #print("----")
            episode.media.pop(f"tmp_{Postprocessing_Custom.ALLOCATION}", None)
        """ or None

        list_relevant_keys_mean = [Postprocessing_Custom.UNWEIGHTED_PENALTY_VIOLATION_SCORE,
                                   Postprocessing_Custom.AMOUNT_CONSTRAINT_VIOLATIONS,
                                  Postprocessing_Custom.BOOL_ANY_CONSTRAINT_VIOLATIONS
        ]

        list_relevant_keys_sum = [Postprocessing_Custom.REWARD_BASE_NO_PENALTIES]

        for key in list_relevant_keys_mean:
            if f"tmp_{key}" in episode.custom_metrics:
                if key not in episode.custom_metrics:
                    episode.custom_metrics[key] = []
                episode.custom_metrics[key].append(np.mean(episode.custom_metrics[f"tmp_{key}"]))
                # removing key
                episode.custom_metrics.pop(f"tmp_{key}", None)

        for key in list_relevant_keys_sum:
            if f"tmp_{key}" in episode.custom_metrics:
                if key not in episode.custom_metrics:
                    episode.custom_metrics[key] = []
                episode.custom_metrics[key].append(np.sum(episode.custom_metrics[f"tmp_{key}"]))
                # removing key
                episode.custom_metrics.pop(f"tmp_{key}", None)


        #Specical Treatment
        if f'tmp_{Postprocessing_Custom.ALLOCATION}' in episode.media:
            allocation_full = np.array(episode.media.pop(f"tmp_{Postprocessing_Custom.ALLOCATION}", None))#[f'tmp_{Postprocessing_Custom.ALLOCATION}'])

            if Postprocessing_Custom.ALLOCATION_mean not in episode.media:
                episode.media[Postprocessing_Custom.ALLOCATION_mean] = []
            episode.media[Postprocessing_Custom.ALLOCATION_mean].append(np.mean(allocation_full, axis=0))
            if Postprocessing_Custom.ALLOCATION_max not in episode.media:
                episode.media[Postprocessing_Custom.ALLOCATION_max] = []
            episode.media[Postprocessing_Custom.ALLOCATION_max].append(np.max(allocation_full, axis=0))
            if Postprocessing_Custom.ALLOCATION_min not in episode.media:
                episode.media[Postprocessing_Custom.ALLOCATION_min] = []
            episode.media[Postprocessing_Custom.ALLOCATION_min].append(np.min(allocation_full, axis=0))

        list_relevant_keys_media_non_aggregation = [Postprocessing_Custom.ACTION_VIOLATIONS,
                                                    Postprocessing_Custom.ACTION_VIOLATIONS_CONSTRAINT_VIOLATIONS]

        for key in list_relevant_keys_media_non_aggregation:
            if f"tmp_{key}" in episode.media:
                if key not in episode.media:
                    episode.media[f"{key}"] = []
                episode.media[f"{key}"].append(episode.media.pop(f"tmp_{key}", None))
        #Special Treatment
        #if f'tmp_{Postprocessing_Custom.ALLOCATION}' in episode.media:

        #List risk relevant
        worker_config = worker.get_policy().config
        if "custom_model_config" in worker_config.get("model") and "config_moment_model" in worker_config.get("model").get(
                "custom_model_config"):

            list_relevant_keys = [Postprocessing_Custom.MSE_FIRST_MOMENT, Postprocessing_Custom.MSE_SECOND_MOMENT,
                                  Postprocessing_Custom.RISK_PENALTY, Postprocessing_Custom.REWARDS_INCL_RISK,
                                  Postprocessing_Custom.REWARDS_BASE_WITHOUT_RISK_PENALTY
                                  ]

            for key in list_relevant_keys:
                if f"tmp_{key}" in episode.custom_metrics:
                    if key not in episode.custom_metrics:
                        episode.custom_metrics[key] = []
                    episode.custom_metrics[key].append(np.sum(episode.custom_metrics[f"tmp_{key}"]))
                    # removing key
                    episode.custom_metrics.pop(f"tmp_{key}", None)

            list_relevant_keys_no_aggregation = [
                #Postprocessing_Custom.DEBUG_MOMENTS,
                Postprocessing_Custom.RISK_VARIANCE_ESTIMATE,
                #SampleBatch.ACTIONS
            ]

            for key in list_relevant_keys_no_aggregation:
                if f"tmp_{key}" in episode.custom_metrics:
                    if key not in episode.custom_metrics:
                        episode.custom_metrics[key] = []
                    episode.custom_metrics[key].append(episode.custom_metrics[f"tmp_{key}"])

                    # removing key
                    episode.custom_metrics.pop(f"tmp_{key}", None)


    def on_episode_step(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        episode: Union[Episode, EpisodeV2],
        **kwargs,
    ) -> None:
        """Runs on each episode step.
        Args:
            worker: Reference to the current rollout worker.
            base_env: BaseEnv running the episode. The underlying
                sub environment objects can be retrieved by calling
                `base_env.get_sub_environments()`.
            policies: Mapping of policy id to policy objects.
                In single agent mode there will only be a single
                "default_policy".
            episode: Episode object which contains episode
                state. You can use the `episode.user_data` dict to store
                temporary data, and `episode.custom_metrics` to store custom
                metrics for the episode.
            kwargs: Forward compatibility placeholder.
        """

        env_config = worker.get_policy().config.get("env_config")

        #last_allocation = self.calculate_action_allocation(episode, env_config)
        last_allocation = calculate_action_allocation(episode.last_action_for(), env_config, worker.get_policy().model)

        key = Postprocessing_Custom.ALLOCATION
        if f"tmp_{key}" not in episode.media:
            episode.media[f"tmp_{key}"] = []
        episode.media[f"tmp_{key}"].append(last_allocation)

    """
    def calculate_action_allocation(self, episode, env_config):

        self.force_single_simplex = env_config.get("force_single_simplex", False)
        self.force_box_space = env_config.get("force_box_space", False)
        np_raw_action = episode.last_action_for()
        if not self.force_single_simplex and not self.force_box_space and "head_factor_list" in env_config:
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
            # return normal last action
            return episode.last_action_for()
    """ or None

    """
    def check_constraints_satisfaction(self, env_config):

        if "head_factor_list" in env_config:
            list_head_factor = env_config.get("head_factor_list")
        if "action_mask_dict" in env_config:#env_config
            env_config.get("action_mask_dict")

    def build_constraint_sets(self, env_config):

        dict_condition = {}
        #WITHOUT TRAINABLE FOR NOW

        if "head_factor_list" in env_config:
            list_head_factor = env_config.get("head_factor_list")
        if "action_mask_dict" in env_config:  # env_config
            action_mask_dict = env_config.get("action_mask_dict")


        list_constraints = []
        # generating all constraints
        for idx, head_factor in enumerate(list_head_factor):

            tmp_condition = {}
            tmp_np_action_mask = np.array(action_mask_dict.get(f'{idx}_action_mask'))
            #tmp_condition["action_mask"] = tmp_action_mask
            #print(tmp_action_mask)
            tmp_tuple_indices = np.where(tmp_np_action_mask==1)
            assert len(tmp_tuple_indices)==1
            tmp_list_indices = tmp_tuple_indices[0].tolist()
            #print(tmp_list_indices)
            tmp_condition["constraint_index"] = idx
            tmp_condition["constraint_variable_indices"] = tmp_list_indices
            tmp_condition["constraint_value"] = head_factor
            tmp_condition["np_action_mask"] = tmp_np_action_mask
            #print(tmp_condition)
            list_constraints.append(tmp_condition)
            #dict_condition[]
            #np.where(arr == 15)
        print(list_constraints)
        print(len(list_constraints))
        #evaluating the relations of all constraints towards each other
        for idx_base, constraint_base in enumerate(list_constraints):
            tmp_constraint_list = list_constraints.copy()
            #tmp_constraint_list.pop(idx_base)
            #store indices
            constraint_base["set_relations"] = {
                "h_plus": [],
                "h_minus": [],
                "k_plus": [],
                "k_minus": []
            }
            for compared_constraint in tmp_constraint_list:
                #*2 to ensure another encoding
                tmp_sum_array = constraint_base.get("np_action_mask")+compared_constraint.get("np_action_mask")*2
                tmp_set_sums = set(tmp_sum_array.tolist()) #all events
                if 3 in tmp_set_sums and 2 in tmp_set_sums and 1 in tmp_set_sums:
                    raise ValueError('Unspecified case')
                elif 3 in tmp_set_sums and 2 in tmp_set_sums:
                    # dominant set -> other set > base set -> base set is smaller than the new set
                    if compared_constraint.get("constraint_value") >= 0:
                        constraint_base.get("set_relations").get("k_plus").append(
                            compared_constraint.get("constraint_index"))
                    else:
                        constraint_base.get("set_relations").get("k_minus").append(compared_constraint.get("constraint_index"))
                elif 3 in tmp_set_sums and 1 in tmp_set_sums:
                    # dominanted set -> base set > other set -> we are smaller than the base set
                    if compared_constraint.get("constraint_value") >= 0:
                        constraint_base.get("set_relations").get("h_plus").append(compared_constraint.get("constraint_index"))
                    else:
                        constraint_base.get("set_relations").get("h_minus").append(compared_constraint.get("constraint_index"))
                elif 3 in tmp_set_sums:
                    # perfect overlap
                    if compared_constraint.get("constraint_value") >= 0:
                        constraint_base.get("set_relations").get("h_plus").append(compared_constraint.get("constraint_index"))
                    else:
                        constraint_base.get("set_relations").get("h_minus").append(compared_constraint.get("constraint_index"))
                elif 2 in tmp_set_sums and 1 in tmp_set_sums:
                    pass # perfectly disjoint -> no influence whatsoever
                else:
                    raise ValueError('Unknown case')
                #[1,0,1,0,0]
                #[2,0,2,0,0] If just 3 and/or 0 -> Perfect overlap
                #[0,2,0,2,0] If just 2, 1 and/or 0 -> perfect disjoint
                #[2,2,0,0,0] 3/2/1 and/or 0 -> unknown case
                #[2,2,2,2,2] 3/2 and/or 0 -> dominant set
                #[2,0,0,0, 0] 3/1 and/or 0 -> dominated set
            print(constraint_base)

        #0
        #_action_mask
    """ or None