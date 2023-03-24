from ray.rllib.models.tf.tf_action_dist import Categorical, ActionDistribution
from ray.rllib.models.torch.torch_action_dist import TorchCategorical, \
    TorchDistributionWrapper
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.annotations import override, DeveloperAPI
from ray.rllib.utils.typing import TensorType

from dirichlet_custom import TorchDirichlet_Custom
import numpy as np

from itertools import accumulate

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()


class TorchAutoregressiveDirichletDistributionS3(TorchDistributionWrapper):

    def __init__(self, inputs, model):
        """Input is a tensor of logits. The exponential of logits is used to
        parametrize the Dirichlet distribution as all parameters need to be
        positive. An arbitrary small epsilon is added to the concentration
        parameters to be zero due to numerical error.
        See issue #4440 for more details.
        We do not use inherentece here for code readability
        """
        self.epsilon = torch.tensor(1e-7).to(inputs.device)
        dummy_concentration = torch.exp(inputs) + self.epsilon

        self.dist_a1 = None
        self.last_sample_a1 = None
        self.dist_a2 = None
        self.last_sample_a2 = None
        self.dist_a3 = None
        self.last_sample_a3 = None

        # we have to run the sampling already here to determine the entire distribution
        super().__init__(dummy_concentration, model)


        #setting parameters by sampling already here
        a1_logit = self.model.forward_action_model_a1(self.inputs)

        concentration_a1 = torch.exp(a1_logit) + self.epsilon

        self.dist_a1 = torch.distributions.dirichlet.Dirichlet(
            concentration=concentration_a1,
            validate_args=True,
        )
        self.last_sample_a1 = self.dist_a1.sample()

        a2_logit = self.model.forward_action_model_a2(self.inputs, self.last_sample_a1)
        concentration_a2 = torch.exp(a2_logit) + self.epsilon

        self.dist_a2 = torch.distributions.dirichlet.Dirichlet(
            concentration=concentration_a2,
            validate_args=True,
        )
        self.last_sample_a2 = self.dist_a2.sample()

        a3_logit = self.model.forward_action_model_a3(self.inputs, self.last_sample_a1, self.last_sample_a2)
        concentration_a3 = torch.exp(a3_logit) + self.epsilon

        self.dist_a3 = torch.distributions.dirichlet.Dirichlet(
            concentration=concentration_a3,
            validate_args=True,
        )

        self.last_sample_a3 = self.dist_a3.sample()

        self.last_sample = torch.cat([self.last_sample_a1, self.last_sample_a2, self.last_sample_a3], 1)

        self.called_sample_already=False

    @override(ActionDistribution)
    def deterministic_sample(self) -> TensorType:

        epsilon = torch.tensor(1e-7).to(self.inputs.device)
        a1_logit = self.model.forward_action_model_a1(self.inputs)
        concentration_a1 = torch.exp(a1_logit) + epsilon
        self.last_sample_a1 = nn.functional.softmax(concentration_a1, dim=1)

        a2_logit = self.model.forward_action_model_a2(self.inputs, self.last_sample_a1)
        concentration_a2 = torch.exp(a2_logit) + epsilon
        self.last_sample_a2 = nn.functional.softmax(concentration_a2, dim=1)

        a3_logit = self.model.forward_action_model_a3(self.inputs, self.last_sample_a1, self.last_sample_a2)
        concentration_a3 = torch.exp(a3_logit) + epsilon
        self.last_sample_a3 = nn.functional.softmax(concentration_a3, dim=1)

        self.last_sample = torch.cat([self.last_sample_a1, self.last_sample_a2, self.last_sample_a3], 1)
        return self.action_output_wrapper(self.last_sample)

    @override(ActionDistribution)
    def sample(self):

        assert not self.called_sample_already
        self.called_sample_already=True

        return self.action_output_wrapper(self.last_sample)

    @override(ActionDistribution)
    def logp(self, action):
        # Support of Dirichlet are positive real numbers. x is already
        # an array of positive numbers, but we clip to avoid zeros due to
        # numerical errors.

        if self.dist_a1 is None or self.dist_a2 is None:
            raise ValueError(f'dist_a1 or dist_a2 should not be None')

        #slice actions into different aspects
        list_actions = self.slice_action(action)
        action_a1 = list_actions[0]
        action_a2 = list_actions[1]
        action_a3 = list_actions[2]

        #action a_1
        epsilon_a1 = torch.tensor(1e-7).to(self.inputs.device)
        action_a1 = torch.max(action_a1, epsilon_a1)
        action_a1 = action_a1 / torch.sum(action_a1, dim=-1, keepdim=True)

        #action a_2
        epsilon_a2 = torch.tensor(1e-7).to(self.inputs.device)
        action_a2 = torch.max(action_a2, epsilon_a2)
        action_a2 = action_a2 / torch.sum(action_a2, dim=-1, keepdim=True)

        #action a_3
        epsilon_a3 = torch.tensor(1e-7).to(self.inputs.device)
        action_a3 = torch.max(action_a3, epsilon_a3)
        action_a3 = action_a3 / torch.sum(action_a3, dim=-1, keepdim=True)

        return self.dist_a1.log_prob(action_a1)+self.dist_a2.log_prob(action_a2)+self.dist_a3.log_prob(action_a3)

    @override(ActionDistribution)
    def entropy(self):
        return self.dist_a1.entropy()+self.dist_a2.entropy()+self.dist_a3.entropy()

    @override(ActionDistribution)
    def kl(self, other):
        kl_dist1 = torch.distributions.kl.kl_divergence(self.dist_a1, other.dist_a1)
        kl_dist2 = torch.distributions.kl.kl_divergence(self.dist_a2, other.dist_a2)
        kl_dist3 = torch.distributions.kl.kl_divergence(self.dist_a3, other.dist_a3)
        return kl_dist1 + kl_dist2 + kl_dist3

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return model_config.get("fcnet_hiddens")[-1]

    def action_output_wrapper(self, action):
        list_actions = self.slice_action(action)
        return {'0_allocation': list_actions[0], '1_allocation': list_actions[1], '2_allocation': list_actions[2]}

    def slice_action(self, action):

        list_sub_actions = torch.split(action, split_size_or_sections=list(self.model.action_space_dim_dict.values()),
                                       dim=1)
        return list_sub_actions



class TorchAutoregressiveDirichletDistributionS2(TorchDistributionWrapper):

    def __init__(self, inputs, model):
        """Input is a tensor of logits. The exponential of logits is used to
        parametrize the Dirichlet distribution as all parameters need to be
        positive. An arbitrary small epsilon is added to the concentration
        parameters to be zero due to numerical error.
        See issue #4440 for more details.
        """
        self.epsilon = torch.tensor(1e-7).to(inputs.device)
        dummy_concentration = torch.exp(inputs) + self.epsilon

        self.dist_a1 = None
        self.last_sample_a1 = None
        self.dist_a2 = None
        self.last_sample_a2 = None

        # we have to run the sampling already here to determine the entire distribution
        super().__init__(dummy_concentration, model)


        #setting parameters by sampling already here
        a1_logit = self.model.forward_action_model_a1(self.inputs)

        concentration_a1 = torch.exp(a1_logit) + self.epsilon

        self.dist_a1 = torch.distributions.dirichlet.Dirichlet(
            concentration=concentration_a1,
            validate_args=True,
        )
        self.last_sample_a1 = self.dist_a1.sample()

        a2_logit = self.model.forward_action_model_a2(self.inputs, self.last_sample_a1)
        concentration_a2 = torch.exp(a2_logit) + self.epsilon

        self.dist_a2 = torch.distributions.dirichlet.Dirichlet(
            concentration=concentration_a2,
            validate_args=True,
        )
        self.last_sample_a2 = self.dist_a2.sample()

        self.last_sample = torch.cat([self.last_sample_a1, self.last_sample_a2], 1)

        self.called_sample_already=False

    @override(ActionDistribution)
    def deterministic_sample(self) -> TensorType:

        epsilon = torch.tensor(1e-7).to(self.inputs.device)
        a1_logit = self.model.forward_action_model_a1(self.inputs)
        concentration_a1 = torch.exp(a1_logit) + epsilon
        self.last_sample_a1 = nn.functional.softmax(concentration_a1, dim=1)

        a2_logit = self.model.forward_action_model_a2(self.inputs, self.last_sample_a1)
        concentration_a2 = torch.exp(a2_logit) + epsilon
        self.last_sample_a2 = nn.functional.softmax(concentration_a2, dim=1)

        self.last_sample = torch.cat([self.last_sample_a1, self.last_sample_a2], 1)
        return self.action_output_wrapper(self.last_sample)

    @override(ActionDistribution)
    def sample(self):

        assert not self.called_sample_already
        self.called_sample_already=True

        return self.action_output_wrapper(self.last_sample)

    @override(ActionDistribution)
    def logp(self, action):
        # Support of Dirichlet are positive real numbers. x is already
        # an array of positive numbers, but we clip to avoid zeros due to
        # numerical errors.

        if self.dist_a1 is None or self.dist_a2 is None:
            raise ValueError(f'dist_a1 or dist_a2 should not be None')

        #slice actions into different aspects
        list_actions = self.slice_action(action)
        action_a1 = list_actions[0]
        action_a2 = list_actions[1]

        #action a_1
        epsilon_a1 = torch.tensor(1e-7).to(self.inputs.device)
        action_a1 = torch.max(action_a1, epsilon_a1)
        action_a1 = action_a1 / torch.sum(action_a1, dim=-1, keepdim=True)

        #action a_2
        epsilon_a2 = torch.tensor(1e-7).to(self.inputs.device)
        action_a2 = torch.max(action_a2, epsilon_a2)
        action_a2 = action_a2 / torch.sum(action_a2, dim=-1, keepdim=True)

        return self.dist_a1.log_prob(action_a1)+self.dist_a2.log_prob(action_a2)

    @override(ActionDistribution)
    def entropy(self):
        return self.dist_a1.entropy()+self.dist_a2.entropy()

    @override(ActionDistribution)
    def kl(self, other):
        kl_dist1 = torch.distributions.kl.kl_divergence(self.dist_a1, other.dist_a1)
        kl_dist2 = torch.distributions.kl.kl_divergence(self.dist_a2, other.dist_a2)
        return kl_dist1 + kl_dist2

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return model_config.get("fcnet_hiddens")[-1]

    def action_output_wrapper(self, action):
        list_actions = self.slice_action(action)
        return {'0_allocation': list_actions[0], '1_allocation': list_actions[1]}

    def slice_action(self, action):

        list_sub_actions = torch.split(action, split_size_or_sections=list(self.model.action_space_dim_dict.values()),
                                       dim=1)
        return list_sub_actions



class TorchAutoregressiveDirichletDistributionS4(TorchDistributionWrapper):

    def __init__(self, inputs, model):
        """Input is a tensor of logits. The exponential of logits is used to
        parametrize the Dirichlet distribution as all parameters need to be
        positive. An arbitrary small epsilon is added to the concentration
        parameters to be zero due to numerical error.
        See issue #4440 for more details.
        We do not use inherentece here for code readability
        """
        self.epsilon = torch.tensor(1e-7).to(inputs.device)
        dummy_concentration = torch.exp(inputs) + self.epsilon

        self.dist_a1 = None
        self.last_sample_a1 = None
        self.dist_a2 = None
        self.last_sample_a2 = None
        self.dist_a3 = None
        self.last_sample_a3 = None
        self.dist_a4 = None
        self.last_sample_a4 = None

        # we have to run the sampling already here to determine the entire distribution
        super().__init__(dummy_concentration, model)


        #setting parameters by sampling already here
        a1_logit = self.model.forward_action_model_a1(self.inputs)

        concentration_a1 = torch.exp(a1_logit) + self.epsilon

        self.dist_a1 = torch.distributions.dirichlet.Dirichlet(
            concentration=concentration_a1,
            validate_args=True,
        )
        self.last_sample_a1 = self.dist_a1.sample()
        ###
        a2_logit = self.model.forward_action_model_a2(self.inputs, self.last_sample_a1)
        concentration_a2 = torch.exp(a2_logit) + self.epsilon

        self.dist_a2 = torch.distributions.dirichlet.Dirichlet(
            concentration=concentration_a2,
            validate_args=True,
        )
        self.last_sample_a2 = self.dist_a2.sample()
        ###
        a3_logit = self.model.forward_action_model_a3(self.inputs, self.last_sample_a1, self.last_sample_a2)
        concentration_a3 = torch.exp(a3_logit) + self.epsilon

        self.dist_a3 = torch.distributions.dirichlet.Dirichlet(
            concentration=concentration_a3,
            validate_args=True,
        )

        self.last_sample_a3 = self.dist_a3.sample()
        ###
        a4_logit = self.model.forward_action_model_a4(self.inputs, self.last_sample_a1, self.last_sample_a2,
                                                      self.last_sample_a3)
        concentration_a4 = torch.exp(a4_logit) + self.epsilon

        self.dist_a4 = torch.distributions.dirichlet.Dirichlet(
            concentration=concentration_a4,
            validate_args=True,
        )

        self.last_sample_a4 = self.dist_a4.sample()
        ###
        self.last_sample = torch.cat([self.last_sample_a1, self.last_sample_a2, self.last_sample_a3,
                                      self.last_sample_a4], 1)

        self.called_sample_already=False

    @override(ActionDistribution)
    def deterministic_sample(self) -> TensorType:

        epsilon = torch.tensor(1e-7).to(self.inputs.device)
        a1_logit = self.model.forward_action_model_a1(self.inputs)
        concentration_a1 = torch.exp(a1_logit) + epsilon
        self.last_sample_a1 = nn.functional.softmax(concentration_a1, dim=1)

        a2_logit = self.model.forward_action_model_a2(self.inputs, self.last_sample_a1)
        concentration_a2 = torch.exp(a2_logit) + epsilon
        self.last_sample_a2 = nn.functional.softmax(concentration_a2, dim=1)

        a3_logit = self.model.forward_action_model_a3(self.inputs, self.last_sample_a1, self.last_sample_a2)
        concentration_a3 = torch.exp(a3_logit) + epsilon
        self.last_sample_a3 = nn.functional.softmax(concentration_a3, dim=1)

        a4_logit = self.model.forward_action_model_a4(self.inputs, self.last_sample_a1, self.last_sample_a2,
                                                      self.last_sample_a3)
        concentration_a4 = torch.exp(a4_logit) + epsilon
        self.last_sample_a4 = nn.functional.softmax(concentration_a4, dim=1)

        self.last_sample = torch.cat([self.last_sample_a1, self.last_sample_a2, self.last_sample_a3,
                                      self.last_sample_a4], 1)
        return self.action_output_wrapper(self.last_sample)

    @override(ActionDistribution)
    def sample(self):

        assert not self.called_sample_already
        self.called_sample_already=True

        return self.action_output_wrapper(self.last_sample)

    @override(ActionDistribution)
    def logp(self, action):
        # Support of Dirichlet are positive real numbers. x is already
        # an array of positive numbers, but we clip to avoid zeros due to
        # numerical errors.

        if self.dist_a1 is None or self.dist_a2 is None or self.dist_a3 is None or self.dist_a4 is None:
            raise ValueError(f'dist_a1 or dist_a2 should not be None')

        #slice actions into different aspects
        list_actions = self.slice_action(action)
        action_a1 = list_actions[0]
        action_a2 = list_actions[1]
        action_a3 = list_actions[2]
        action_a4 = list_actions[3]

        #action a_1
        epsilon_a1 = torch.tensor(1e-7).to(self.inputs.device)
        action_a1 = torch.max(action_a1, epsilon_a1)
        action_a1 = action_a1 / torch.sum(action_a1, dim=-1, keepdim=True)

        #action a_2
        epsilon_a2 = torch.tensor(1e-7).to(self.inputs.device)
        action_a2 = torch.max(action_a2, epsilon_a2)
        action_a2 = action_a2 / torch.sum(action_a2, dim=-1, keepdim=True)

        #action a_3
        epsilon_a3 = torch.tensor(1e-7).to(self.inputs.device)
        action_a3 = torch.max(action_a3, epsilon_a3)
        action_a3 = action_a3 / torch.sum(action_a3, dim=-1, keepdim=True)

        # action a_4
        epsilon_a4 = torch.tensor(1e-7).to(self.inputs.device)
        action_a4 = torch.max(action_a4, epsilon_a4)
        action_a4 = action_a4 / torch.sum(action_a4, dim=-1, keepdim=True)

        return self.dist_a1.log_prob(action_a1)+self.dist_a2.log_prob(action_a2)+self.dist_a3.log_prob(action_a3)+\
               self.dist_a4.log_prob(action_a4)

    @override(ActionDistribution)
    def entropy(self):
        return self.dist_a1.entropy()+self.dist_a2.entropy()+self.dist_a3.entropy()+self.dist_a4.entropy()

    @override(ActionDistribution)
    def kl(self, other):
        kl_dist1 = torch.distributions.kl.kl_divergence(self.dist_a1, other.dist_a1)
        kl_dist2 = torch.distributions.kl.kl_divergence(self.dist_a2, other.dist_a2)
        kl_dist3 = torch.distributions.kl.kl_divergence(self.dist_a3, other.dist_a3)
        kl_dist4 = torch.distributions.kl.kl_divergence(self.dist_a4, other.dist_a4)
        return kl_dist1 + kl_dist2 + kl_dist3 + kl_dist4

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return model_config.get("fcnet_hiddens")[-1]

    def action_output_wrapper(self, action):
        list_actions = self.slice_action(action)
        return {'0_allocation': list_actions[0], '1_allocation': list_actions[1], '2_allocation': list_actions[2],
                '3_allocation': list_actions[3]}

    def slice_action(self, action):

        list_sub_actions = torch.split(action, split_size_or_sections=list(self.model.action_space_dim_dict.values()),
                                       dim=1)
        return list_sub_actions



class TorchAutoregressiveDirichletDistributionS4_U1(TorchDistributionWrapper):

    def __init__(self, inputs, model):
        """Input is a tensor of logits. The exponential of logits is used to
        parametrize the Dirichlet distribution as all parameters need to be
        positive. An arbitrary small epsilon is added to the concentration
        parameters to be zero due to numerical error.
        See issue #4440 for more details.
        We do not use inherentece here for code readability
        """
        self.epsilon = torch.tensor(1e-7).to(inputs.device)
        dummy_concentration = torch.exp(inputs) + self.epsilon


        self.dist_u1 = None
        self.last_sample_u1 = None
        self.dist_a1 = None
        self.last_sample_a1 = None
        self.dist_a2 = None
        self.last_sample_a2 = None
        self.dist_a3 = None
        self.last_sample_a3 = None
        self.dist_a4 = None
        self.last_sample_a4 = None

        # we have to run the sampling already here to determine the entire distribution
        super().__init__(dummy_concentration, model)

        ###
        u1_logit = self.model.forward_action_model_u1(self.inputs)

        list_u1_logit = torch.tensor_split(u1_logit, 2, dim=1)

        concentration_u1_1 = torch.exp(list_u1_logit[0]) #to ensure alpha>0
        concentration_u1_2 = torch.exp(list_u1_logit[1]) #to ensure beta>0

        self.dist_u1 = torch.distributions.beta.Beta(concentration_u1_1, concentration_u1_2)
        self.last_sample_u1 = self.dist_u1.sample()

        #setting parameters by sampling already here
        a1_logit = self.model.forward_action_model_a1(self.inputs, self.last_sample_u1)

        concentration_a1 = torch.exp(a1_logit) + self.epsilon

        self.dist_a1 = torch.distributions.dirichlet.Dirichlet(
            concentration=concentration_a1,
            validate_args=True,
        )
        self.last_sample_a1 = self.dist_a1.sample()
        ###
        a2_logit = self.model.forward_action_model_a2(self.inputs, self.last_sample_u1, self.last_sample_a1)
        concentration_a2 = torch.exp(a2_logit) + self.epsilon

        self.dist_a2 = torch.distributions.dirichlet.Dirichlet(
            concentration=concentration_a2,
            validate_args=True,
        )
        self.last_sample_a2 = self.dist_a2.sample()
        ###
        a3_logit = self.model.forward_action_model_a3(self.inputs, self.last_sample_u1, self.last_sample_a1, self.last_sample_a2)
        concentration_a3 = torch.exp(a3_logit) + self.epsilon

        self.dist_a3 = torch.distributions.dirichlet.Dirichlet(
            concentration=concentration_a3,
            validate_args=True,
        )

        self.last_sample_a3 = self.dist_a3.sample()
        ###
        a4_logit = self.model.forward_action_model_a4(self.inputs, self.last_sample_u1, self.last_sample_a1, self.last_sample_a2,
                                                      self.last_sample_a3)
        concentration_a4 = torch.exp(a4_logit) + self.epsilon

        self.dist_a4 = torch.distributions.dirichlet.Dirichlet(
            concentration=concentration_a4,
            validate_args=True,
        )

        self.last_sample_a4 = self.dist_a4.sample()
        ###
        #self.last_sample = torch.cat([self.last_sample_u1, self.last_sample_a1, self.last_sample_a2, self.last_sample_a3,
         #                             self.last_sample_a4], 1)
        self.last_sample = TorchAutoregressiveDirichletDistributionS4_U1.merge_action(self.model,
            action_u1=self.last_sample_u1, action_a1=self.last_sample_a1,
                                             action_a2=self.last_sample_a2, action_a3=self.last_sample_a3,
                                             action_a4=self.last_sample_a4)

        self.called_sample_already=False

    @override(ActionDistribution)
    def deterministic_sample(self) -> TensorType:
        u1_logit = self.model.forward_action_model_u1(self.inputs)
        list_u1_logit = torch.tensor_split(u1_logit, 2, dim=1)
        concentration_u1_1 = torch.exp(list_u1_logit[0])  # to ensure alpha>0
        concentration_u1_2 = torch.exp(list_u1_logit[1])  # to ensure beta>0
        self.last_sample_u1 = concentration_u1_1/(concentration_u1_1+concentration_u1_2)

        epsilon = torch.tensor(1e-7).to(self.inputs.device)
        a1_logit = self.model.forward_action_model_a1(self.inputs, self.last_sample_u1)
        concentration_a1 = torch.exp(a1_logit) + epsilon
        self.last_sample_a1 = nn.functional.softmax(concentration_a1, dim=1)

        a2_logit = self.model.forward_action_model_a2(self.inputs, self.last_sample_u1, self.last_sample_a1)
        concentration_a2 = torch.exp(a2_logit) + epsilon
        self.last_sample_a2 = nn.functional.softmax(concentration_a2, dim=1)

        a3_logit = self.model.forward_action_model_a3(self.inputs, self.last_sample_u1, self.last_sample_a1, self.last_sample_a2)
        concentration_a3 = torch.exp(a3_logit) + epsilon
        self.last_sample_a3 = nn.functional.softmax(concentration_a3, dim=1)

        a4_logit = self.model.forward_action_model_a4(self.inputs, self.last_sample_u1, self.last_sample_a1, self.last_sample_a2,
                                                      self.last_sample_a3)
        concentration_a4 = torch.exp(a4_logit) + epsilon
        self.last_sample_a4 = nn.functional.softmax(concentration_a4, dim=1)

        self.last_sample = TorchAutoregressiveDirichletDistributionS4_U1.merge_action(self.model,
            action_u1=self.last_sample_u1, action_a1=self.last_sample_a1,
                                             action_a2=self.last_sample_a2, action_a3=self.last_sample_a3,
                                             action_a4=self.last_sample_a4)
        #self.last_sample = torch.cat([self.last_sample_u1, self.last_sample_a1, self.last_sample_a2, self.last_sample_a3,
                           #           self.last_sample_a4], 1)
        return TorchAutoregressiveDirichletDistributionS4_U1.action_output_wrapper(self.model, self.last_sample)

    @override(ActionDistribution)
    def sample(self):

        assert not self.called_sample_already
        self.called_sample_already=True

        return TorchAutoregressiveDirichletDistributionS4_U1.action_output_wrapper(self.model, self.last_sample)

    @override(ActionDistribution)
    def logp(self, action):
        # Support of Dirichlet are positive real numbers. x is already
        # an array of positive numbers, but we clip to avoid zeros due to
        # numerical errors.

        if self.dist_u1 is None or self.dist_a1 is None or self.dist_a2 is None or self.dist_a3 is None or self.dist_a4 is None:
            raise ValueError(f'dist_a1 or dist_a2 should not be None')

        #slice actions into different aspects
        #list_actions = self.slice_action(action)
        dict_actions = TorchAutoregressiveDirichletDistributionS4_U1.action_output_wrapper(self.model, action)
        action_u1 = dict_actions.get("0_uniform_factor")
        action_a1 = dict_actions.get("0_allocation")
        action_a2 = dict_actions.get("1_allocation")
        action_a3 = dict_actions.get("2_allocation")
        action_a4 = dict_actions.get("3_allocation")

        #action u_1
        epsilon_u1 = torch.tensor(1e-7).to(self.inputs.device)
        action_u1 = torch.max(action_u1, epsilon_u1)

        #action a_1
        epsilon_a1 = torch.tensor(1e-7).to(self.inputs.device)
        action_a1 = torch.max(action_a1, epsilon_a1)
        action_a1 = action_a1 / torch.sum(action_a1, dim=-1, keepdim=True)

        #action a_2
        epsilon_a2 = torch.tensor(1e-7).to(self.inputs.device)
        action_a2 = torch.max(action_a2, epsilon_a2)
        action_a2 = action_a2 / torch.sum(action_a2, dim=-1, keepdim=True)

        #action a_3
        epsilon_a3 = torch.tensor(1e-7).to(self.inputs.device)
        action_a3 = torch.max(action_a3, epsilon_a3)
        action_a3 = action_a3 / torch.sum(action_a3, dim=-1, keepdim=True)

        # action a_4
        epsilon_a4 = torch.tensor(1e-7).to(self.inputs.device)
        action_a4 = torch.max(action_a4, epsilon_a4)
        action_a4 = action_a4 / torch.sum(action_a4, dim=-1, keepdim=True)

        return self.dist_u1.log_prob(action_u1)+self.dist_a1.log_prob(action_a1)+self.dist_a2.log_prob(action_a2)+\
               self.dist_a3.log_prob(action_a3)+self.dist_a4.log_prob(action_a4)

    @override(ActionDistribution)
    def entropy(self):
        return self.dist_u1.entropy()+self.dist_a1.entropy()+self.dist_a2.entropy()+self.dist_a3.entropy()+self.dist_a4.entropy()

    @override(ActionDistribution)
    def kl(self, other):
        kl_dist_u1 = torch.distributions.kl.kl_divergence(self.dist_u1, other.dist_u1)
        kl_dist_a1 = torch.distributions.kl.kl_divergence(self.dist_a1, other.dist_a1)
        kl_dist_a2 = torch.distributions.kl.kl_divergence(self.dist_a2, other.dist_a2)
        kl_dist_a3 = torch.distributions.kl.kl_divergence(self.dist_a3, other.dist_a3)
        kl_dist_a4 = torch.distributions.kl.kl_divergence(self.dist_a4, other.dist_a4)
        return kl_dist_u1 + kl_dist_a1 + kl_dist_a2 + kl_dist_a3 + kl_dist_a4

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return model_config.get("fcnet_hiddens")[-1]

    @staticmethod
    def action_output_wrapper(model, action):
        list_actions = TorchAutoregressiveDirichletDistributionS4_U1.slice_action(model, action)
        dict_action_output = dict(zip(list(model.action_space_dim_dict.keys()), list_actions))
        return dict_action_output
        #return {'0_allocation': list_actions[0], '1_allocation': list_actions[1], '2_allocation': list_actions[2],
        #        '3_allocation': list_actions[3]}

    @staticmethod
    def slice_action(model, action):
        if isinstance(action, torch.Tensor):
            list_sub_actions = torch.split(action, split_size_or_sections=list(model.action_space_dim_dict.values()),
                                       dim=1)
        else: #numpy
            list_index_cumsum = list(accumulate(list(model.action_space_dim_dict.values())))
            if action.ndim == 2:
                list_sub_actions = np.split(action, list_index_cumsum, axis=1)
            elif action.ndim == 1:
                list_sub_actions = np.split(action, list_index_cumsum)
        return list_sub_actions

    @staticmethod
    def merge_action(model, action_u1, action_a1, action_a2, action_a3, action_a4):
        list_actions = []
        for key in model.action_space_dim_dict.keys():
            if key == "0_uniform_factor":
                list_actions.append(action_u1)
            elif key == "0_allocation":
                list_actions.append(action_a1)
            elif key == "1_allocation":
                list_actions.append(action_a2)
            elif key == "2_allocation":
                list_actions.append(action_a3)
            elif key == "3_allocation":
                list_actions.append(action_a4)

        if isinstance(action_u1, torch.Tensor):
            return torch.cat(list_actions, 1)
        else:
            raise NotImplementedError