import logging
from typing import Dict, List, Type, Union

import ray
from ray.rllib.algorithms.ppo.ppo_tf_policy import validate_config
from ray.rllib.evaluation.postprocessing import (
    Postprocessing,
    compute_gae_for_sample_batch,
)
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_mixins import (
    EntropyCoeffSchedule,
    KLCoeffMixin,
    LearningRateSchedule,
    ValueNetworkMixin,
)
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.torch_utils import (
    apply_grad_clipping,
    explained_variance,
    sequence_mask,
)
from ray.rllib.utils.typing import TensorType


torch, nn = try_import_torch()

logger = logging.getLogger(__name__)

from helper_functions import generate_aggregated_constraints, convert_raw_constraints_to_full_constraint_tuple, \
    generate_np_penalty_matrix, check_constraint_violations, generate_list_raw_constraint_tuples, \
    calculate_action_allocation, generate_aggregated_constraints_conditional_minkowski_encoding
from risk_postprocessing import compute_constraint_penalized_rewards, calculate_risk_logic
from helper_function_policy import train_moment_network_attention_model, train_moment_network

def train_lambda_network(policy, lambda_penalty_model, train_batch):
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

    np_action_processed = calculate_action_allocation(train_batch[SampleBatch.ACTIONS].cpu().detach().numpy(),
                                                      policy.config.get("env_config"))

    np_penalty_matrix = generate_np_penalty_matrix(np_action_processed, list_agg_constraints)

    torch_penalty_matrix = torch.from_numpy(np_penalty_matrix).float().to(lambda_penalty_model.availabe_device)
    output = lambda_penalty_model(torch_penalty_matrix)

    loss = torch.sum(
        output)  # turn into max problem (the penalty will grow was big as possible, since we deduct it later)
    lambda_penalty_model.zero_grad()
    #print(f'LOSS: {loss}')
    loss.backward()
    lambda_penalty_model.custom_step()


class RiskLagrangePPOTorchPolicy(
    ValueNetworkMixin,
    LearningRateSchedule,
    EntropyCoeffSchedule,
    KLCoeffMixin,
    TorchPolicyV2,
):
    """PyTorch policy class used with PPO."""

    def __init__(self, observation_space, action_space, config):
        config = dict(ray.rllib.algorithms.ppo.ppo.PPOConfig().to_dict(), **config)
        validate_config(config)

        TorchPolicyV2.__init__(
            self,
            observation_space,
            action_space,
            config,
            max_seq_len=config["model"]["max_seq_len"],
        )

        ValueNetworkMixin.__init__(self, config)
        LearningRateSchedule.__init__(self, config["lr"], config["lr_schedule"])
        EntropyCoeffSchedule.__init__(
            self, config["entropy_coeff"], config["entropy_coeff_schedule"]
        )
        KLCoeffMixin.__init__(self, config)

        # TODO: Don't require users to call this manually.
        self._initialize_loss_from_dummy_batch()

    @override(TorchPolicyV2)
    def loss(
        self,
        model: ModelV2,
        dist_class: Type[ActionDistribution],
        train_batch: SampleBatch,
    ) -> Union[TensorType, List[TensorType]]:
        """Compute loss for Proximal Policy Objective.
        Args:
            model: The Model to calculate the loss for.
            dist_class: The action distr. class.
            train_batch: The training data.
        Returns:
            The PPO loss tensor given the input batch.
        """

        #Insert training moment model
        if "custom_model_config" in self.config.get("model") and "config_moment_model" in self.config.get("model").get("custom_model_config"):
            if model.use_moment_attention:
                first_moment_loss, second_moment_loss = train_moment_network_attention_model(
                    policy=self,
                    moment_model=model.moment_submodel,
                    train_batch=train_batch)
            else:
                first_moment_loss, second_moment_loss = train_moment_network(policy=self,
                                                                             moment_model=model.moment_submodel,
                                                                             train_batch=train_batch,
                                                                             evaluation=False)
        else:
            first_moment_loss, second_moment_loss = (1, 1)  # we set dummy variables

        logits, state = model(train_batch)
        curr_action_dist = dist_class(logits, model)

        #print("----")
        train_lambda_network(policy=self,
                             lambda_penalty_model=model.lambda_penalty_model,
                             train_batch=train_batch)
        #print("----//////////")
        # RNN case: Mask away 0-padded chunks at end of time axis.
        if state:
            B = len(train_batch[SampleBatch.SEQ_LENS])
            max_seq_len = logits.shape[0] // B
            mask = sequence_mask(
                train_batch[SampleBatch.SEQ_LENS],
                max_seq_len,
                time_major=model.is_time_major(),
            )
            mask = torch.reshape(mask, [-1])
            num_valid = torch.sum(mask)

            def reduce_mean_valid(t):
                return torch.sum(t[mask]) / num_valid

        # non-RNN case: No masking.
        else:
            mask = None
            reduce_mean_valid = torch.mean

        prev_action_dist = dist_class(
            train_batch[SampleBatch.ACTION_DIST_INPUTS], model
        )

        logp_ratio = torch.exp(
            curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
            - train_batch[SampleBatch.ACTION_LOGP]
        )

        # Only calculate kl loss if necessary (kl-coeff > 0.0).
        if self.config["kl_coeff"] > 0.0:
            action_kl = prev_action_dist.kl(curr_action_dist)
            mean_kl_loss = reduce_mean_valid(action_kl)
        else:
            mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)

        curr_entropy = curr_action_dist.entropy()
        mean_entropy = reduce_mean_valid(curr_entropy)

        surrogate_loss = torch.min(
            train_batch[Postprocessing.ADVANTAGES] * logp_ratio,
            train_batch[Postprocessing.ADVANTAGES]
            * torch.clamp(
                logp_ratio, 1 - self.config["clip_param"], 1 + self.config["clip_param"]
            ),
        )
        mean_policy_loss = reduce_mean_valid(-surrogate_loss)

        # Compute a value function loss.
        if self.config["use_critic"]:
            value_fn_out = model.value_function()
            vf_loss = torch.pow(
                value_fn_out - train_batch[Postprocessing.VALUE_TARGETS], 2.0
            )
            vf_loss_clipped = torch.clamp(vf_loss, 0, self.config["vf_clip_param"])
            mean_vf_loss = reduce_mean_valid(vf_loss_clipped)
        # Ignore the value function.
        else:
            value_fn_out = 0
            vf_loss_clipped = mean_vf_loss = 0.0

        total_loss = reduce_mean_valid(
            -surrogate_loss
            + self.config["vf_loss_coeff"] * vf_loss_clipped
            - self.entropy_coeff * curr_entropy
        )

        # Add mean_kl_loss (already processed through `reduce_mean_valid`),
        # if necessary.
        if self.config["kl_coeff"] > 0.0:
            total_loss += self.kl_coeff * mean_kl_loss

        # Store values for stats function in model (tower), such that for
        # multi-GPU, we do not override them during the parallel loss phase.
        model.tower_stats["total_loss"] = total_loss
        model.tower_stats["mean_policy_loss"] = mean_policy_loss
        model.tower_stats["mean_vf_loss"] = mean_vf_loss
        model.tower_stats["vf_explained_var"] = explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS], value_fn_out
        )
        model.tower_stats["mean_entropy"] = mean_entropy
        model.tower_stats["mean_kl_loss"] = mean_kl_loss

        #Added
        model.tower_stats["first_moment_loss"] = first_moment_loss
        model.tower_stats["second_moment_loss"] = second_moment_loss

        return total_loss

    # TODO: Make this an event-style subscription (e.g.:
    #  "after_gradients_computed").
    @override(TorchPolicyV2)
    def extra_grad_process(self, local_optimizer, loss):
        return apply_grad_clipping(self, local_optimizer, loss)

    @override(TorchPolicyV2)
    def stats_fn(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        return convert_to_numpy(
            {
                "cur_kl_coeff": self.kl_coeff,
                "cur_lr": self.cur_lr,
                "total_loss": torch.mean(
                    torch.stack(self.get_tower_stats("total_loss"))
                ),
                "policy_loss": torch.mean(
                    torch.stack(self.get_tower_stats("mean_policy_loss"))
                ),
                "vf_loss": torch.mean(
                    torch.stack(self.get_tower_stats("mean_vf_loss"))
                ),
                "vf_explained_var": torch.mean(
                    torch.stack(self.get_tower_stats("vf_explained_var"))
                ),
                "kl": torch.mean(torch.stack(self.get_tower_stats("mean_kl_loss"))),
                "entropy": torch.mean(
                    torch.stack(self.get_tower_stats("mean_entropy"))
                ),
                "entropy_coeff": self.entropy_coeff,
            }
        )

    @override(TorchPolicyV2)
    def postprocess_trajectory(
        self, sample_batch, other_agent_batches=None, episode=None
    ):
        # Do all post-processing always with no_grad().
        # Not using this here will introduce a memory leak
        # in torch (issue #6962).
        # TODO: no_grad still necessary?
        #self.test_func(sample_batch)
        #print(sample_batch[SampleBatch.ACTIONS])
        #print("---")

        with torch.no_grad():

            if "custom_model_config" in self.config.get("model") and "config_moment_model" in self.config.get(
                    "model").get("custom_model_config"):
                sample_batch = calculate_risk_logic(
                    self, sample_batch, other_agent_batches, episode
                )
            #np_penalty_score = calculate_penalty_score(sample_batch=sample_batch,
            #                        lambda_model=self.model.lambda_penalty_model)

            sample_batch = compute_constraint_penalized_rewards(self, sample_batch, other_agent_batches, episode)

            sample_batch = compute_gae_for_sample_batch(
                self, sample_batch, other_agent_batches, episode
            )
            return sample_batch

    @override(TorchPolicyV2)
    def make_model(self):
        from risk_lambda_custom_model import make_risk_model
        return make_risk_model(policy=self,
                               obs_space=self.observation_space,
                               action_space=self.action_space,
                               config=self.config)

    def test_func(self, sample_batch):

        list_raw_constraint_tuples = [
            (0.5, [1, 1, 1, 0, 0]),
            (0.5, [0, 0, 0, 1, 1])
        ]

        list_relationship_enriched_tuple = convert_raw_constraints_to_full_constraint_tuple(
            list_constraint_tuple=list_raw_constraint_tuples)
        list_agg_constraints = generate_aggregated_constraints(list_relationship_enriched_tuple)

        np_penalty_matrix = generate_np_penalty_matrix(sample_batch[SampleBatch.ACTIONS], list_agg_constraints)

        #self.lambda_penalty_model
        #from lambda_penalty_model import LambdaPenaltyModel
        #model = LambdaPenaltyModel(inputSize=2*len(list_raw_constraint_tuples))
        torch_penalty_matrix = torch.from_numpy(np_penalty_matrix).float().to(self.model.lambda_penalty_model.availabe_device)
        output = self.model.lambda_penalty_model(torch_penalty_matrix)

        loss = -torch.sum(output)  # turn into max problem (the penalty will grow was big as possible, since we deduct it later)
        self.model.lambda_penalty_model.zero_grad()
        loss.backward()
        self.model.lambda_penalty_model.custom_step()

        print(output)
        print("~~~")
        #print(np_penalty_matrix)

        amount_violations = check_constraint_violations(np_penalty_matrix)
        #print(amount_violations)
        #print("####")
        # Model will return the overall penalty score (sum_i lambda_i penalty_i) -> which is a scalar: n inputs to 1 output
        # this scalar will be deducted from the