import torch
import numpy as np
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch

from ray.rllib.utils.typing import TensorType

from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.utils.annotations import override, DeveloperAPI


torch, nn = try_import_torch()

# This implementation is just necessary due to the incorrect implementation of the KL distance function for TorchDirichlet
@DeveloperAPI
class TorchDirichlet_Custom(TorchDistributionWrapper):
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

    @override(ActionDistribution)
    def deterministic_sample(self) -> TensorType:
        #removed depreciated warning by adding dim=1
        self.last_sample = nn.functional.softmax(self.dist.concentration, dim=1)
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

    # LEAVE THIS OUT, implementation is incorrect -> just go back to default of superclass
    #@override(ActionDistribution)
    #def kl(self, other):
    #    return self.dist.kl_divergence(other.dist)

    @staticmethod
    @override(ActionDistribution)
    def required_model_output_shape(action_space, model_config):
        return np.prod(action_space.shape)
