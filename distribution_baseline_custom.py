from ray.rllib.models.tf.tf_action_dist import Categorical, ActionDistribution
from ray.rllib.models.torch.torch_action_dist import TorchCategorical, \
    TorchDistributionWrapper
from ray.rllib.utils.framework import try_import_tf, try_import_torch

from ray.rllib.utils.annotations import override, DeveloperAPI, ExperimentalAPI
import numpy as np
from ray.rllib.utils.typing import TensorType, List, Union, Tuple, ModelConfigDict
import gym

from polytope_evaluator import PolytopeEvaluator

from helper_functions import generate_aggregated_constraints_conditional_minkowski_encoding

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

@DeveloperAPI
class TorchBaselinePolytopeDistribution(TorchDistributionWrapper):
    """Dirichlet distribution for continuous actions that are between
    [0,1] and sum to 1.
    e.g. actions that represent resource allocation."""

    def __init__(self, inputs, model):
        """Input is a tensor of logits. The exponential of logits is used to
        parametrize the Dirichlet distribution as all parameters need to be
        positive. An arbitrary small epsilon is added to the concentration
        parameters to be zero due to numerical error.
        See issue #4440 for more details.
        """
        self.epsilon = torch.tensor(1e-7).to(inputs.device)
        concentration = torch.exp(inputs) + self.epsilon
        self.dist = torch.distributions.dirichlet.Dirichlet(
            concentration=concentration,
            validate_args=True,
        )
        super().__init__(concentration, model)

        batch_size = inputs.size()[0]
        self.batch_size = batch_size
        self.input_device = inputs.device
        #list_full_agg_polytope = PolytopeEvaluator.convert_dict_polytope_in_list_full_agg_constraint_tuples(
        #    dict_polytope=model.dict_polytope)

        #model_env_config = policy.config.get("env_config")
        self.model = model

        if self.model.constraints_conditional_minkowski_encoding_type is not None:
            head_factor_list = self.model.dict_polytope.get("head_factor_list")
            action_mask_dict = self.model.dict_polytope.get("action_mask_dict")
            list_agg_constraints = generate_aggregated_constraints_conditional_minkowski_encoding(
                head_factor_list,
                action_mask_dict,
                full_constraint_check=True,
                conditional_minkowski_encoding_type=self.model.constraints_conditional_minkowski_encoding_type)
        else:
            # standard procedure
            # list_raw_constraint_tuples = generate_list_raw_constraint_tuples(model.dict_polytope)

            # list_relationship_enriched_tuple = convert_raw_constraints_to_full_constraint_tuple(
            #    list_constraint_tuple=list_raw_constraint_tuples)
            # list_agg_constraints = generate_aggregated_constraints(list_relationship_enriched_tuple)
            raise NotImplementedError
        print(list_agg_constraints)
        print("~~")
        self.polytope_evaluator = PolytopeEvaluator(list_full_agg_constraint_tuples=list_agg_constraints,
                                                    batch_size=self.batch_size)

        #batch_size = inputs.size()[0]
        #self.list_full_agg_polytope = list_full_agg_polytope
        #self.batch_size = batch_size
        # print(batch_size)

        #self.polytope_evaluator = PolytopeEvaluator(list_full_agg_constraint_tuples=self.list_full_agg_polytope,
        #                                            batch_size=self.batch_size)
        #NOTE IT IS IMPORTANT TO TRANSLATE

    @override(ActionDistribution)
    def sample(self) -> TensorType:
        #self.last_sample = self.dist.sample()

        np_sample = self.polytope_evaluator.sample_complete_polytope_uniformly(number_samples=self.batch_size)

        #print(self.polytope_evaluator.generate_batch_mhar_package_inputs())
        #print("~~~~~~~~")

        self.last_sample = torch.from_numpy(np_sample).float().to(self.input_device)
        #print(self.last_sample)
        #print("~~")

        #self.last_sample = torch.zeros_like(self.last_sample).to(self.input_device)
        #self.last_sample[:, 0] = torch.ones(self.batch_size)
        return self.last_sample

    @override(ActionDistribution)
    def deterministic_sample(self) -> TensorType:
        #self.last_sample = nn.functional.softmax(self.dist.concentration)

        np_sample = self.polytope_evaluator.sample_complete_polytope_uniformly(number_samples=self.batch_size)
        self.last_sample = torch.from_numpy(np_sample).float().to(self.input_device)

        #self.last_sample = torch.zeros_like(self.last_sample).to(self.input_device)
        #self.last_sample[:, 0] = torch.ones(self.batch_size)
        return self.last_sample

    @override(ActionDistribution)
    def logp(self, x):
        # Support of Dirichlet are positive real numbers. x is already
        # an array of positive numbers, but we clip to avoid zeros due to
        # numerical errors.
        x = torch.max(x, self.epsilon)
        x = x / torch.sum(x, dim=-1, keepdim=True)
        return self.dist.log_prob(x)

    @override(ActionDistribution)
    def entropy(self):
        return self.dist.entropy()

    @override(ActionDistribution)
    def kl(self, other):
        return torch.distributions.kl.kl_divergence(self.dist, other.dist)
        #return self.dist.kl_divergence(other.dist)

    @staticmethod
    @override(ActionDistribution)
    def required_model_output_shape(action_space, model_config):
        #if isinstance(action_space, gym.spaces.Dict):
        #    return np.prod(action_space["0_allocation"].shape, dtype=np.int32)
        #else:
        #    raise  ValueError('Please customize for the environment')
        return np.prod(action_space.shape, dtype=np.int32)