from ray.rllib.models.tf.tf_action_dist import Categorical, ActionDistribution
from ray.rllib.models.torch.torch_action_dist import TorchCategorical, \
    TorchDistributionWrapper
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.annotations import override, DeveloperAPI
from ray.rllib.utils.typing import TensorType

from dirichlet_custom import TorchDirichlet_Custom
import numpy as np

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()


class TorchAutoregressiveDirichletDistributionTypeOne(TorchDistributionWrapper):
    """Action distribution P(a1, a2) = P(a1) * P(a2 | a1)"""

    def deterministic_sample(self):
        # First, sample a1.
        a1_dist = self._a1_distribution()
        a1 = a1_dist.deterministic_sample()

        self._action_logp = a1_dist.logp(a1)

        return self.action_output_wrapper(a1)

    def sample(self):
        # First, sample a1.
        a1_dist = self._a1_distribution()
        a1 = a1_dist.sample()

        self._action_logp = a1_dist.logp(a1)

        return self.action_output_wrapper(a1)

    def logp(self, actions):
        #print("LOGP")
        #print(actions)
        #print(self.model.action_space_dim_dict)
        #print("IMPORTANT")
        #a1, a2 = actions[:, 0], actions[:, 1]

        a1 = actions

        #a1_logits, a2_logits = self.model.action_module(self.inputs, a1_vec)
        a1_logits = self.model.forward_action_model(self.inputs)
        #by doing this we safe one calculation
        return TorchDirichlet_Custom(a1_logits, self.model).logp(a1)

    def sampled_action_logp(self):
        return torch.exp(self._action_logp)

    def entropy(self):
        a1_dist = self._a1_distribution()
        return a1_dist.entropy()

    def kl(self, other):
        a1_dist = self._a1_distribution()
        a1_terms = a1_dist.kl(other._a1_distribution())

        return a1_terms

    def _a1_distribution(self):
        BATCH = self.inputs.shape[0]
        #dummies for a2
        #zeros = torch.zeros((BATCH, self.model.action_space_dim_dict.get("a_2"))).to(self.inputs.device)
        a1_logits = self.model.forward_action_model(self.inputs)
        a1_dist = TorchDirichlet_Custom(a1_logits, self.model)
        return a1_dist

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return model_config.get("fcnet_hiddens")[-1] #self.model.hidden_output_size#16  # controls model output feature vector size

    def action_output_wrapper(self, a_1):
        return {'0_allocation': a_1}

    def slice_action(self, action):
        list_sub_actions = torch.split(action, split_size_or_sections=list(self.model.action_space_dim_dict.values()),
                                       dim=1)
        return list_sub_actions

class TorchAutoregressiveDirichletDistributionTypeOneTestingOne(TorchDistributionWrapper):

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
        return self.action_output_wrapper(self.last_sample)

    @override(ActionDistribution)
    def sample(self):
        self.last_sample = self.dist.sample()
        return self.action_output_wrapper(self.last_sample)

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

    def action_output_wrapper(self, a_1):
        return {'0_allocation': a_1}

class TorchAutoregressiveDirichletDistributionTypeOneTestingTwo(TorchDistributionWrapper):

    def _a1_distribution(self):
        BATCH = self.inputs.shape[0]
        #dummies for a2
        #zeros = torch.zeros((BATCH, self.model.action_space_dim_dict.get("a_2"))).to(self.inputs.device)
        #a1_logits = self.model.forward_action_model(self.inputs)
        a1_logits = self.inputs
        a1_dist = TorchDirichlet_Custom(a1_logits, self.model)#TorchAutoregressiveDirichletDistributionTypeOneTestingOne(a1_logits, self.model)
        return a1_dist

    def deterministic_sample(self):
        # First, sample a1.
        a1_dist = self._a1_distribution()
        a1 = a1_dist.deterministic_sample()

        self._action_logp = a1_dist.logp(a1)

        return self.action_output_wrapper(a1)

    def sampled_action_logp(self):
        return torch.exp(self._action_logp)

    def entropy(self):
        a1_dist = self._a1_distribution()
        return a1_dist.entropy()

    def logp(self, actions):
        a1 = actions

        #a1_logits = self.inputs
        a1_dist = self._a1_distribution()#TorchAutoregressiveDirichletDistributionTypeOneTestingOne(a1_logits, self.model)

        #by doing this we safe one calculation
        return a1_dist.logp(a1)

    def sample(self):
        # First, sample a1.
        a1_dist = self._a1_distribution()
        a1 = a1_dist.sample()

        self._action_logp = a1_dist.logp(a1)

        return self.action_output_wrapper(a1)

    def action_output_wrapper(self, a_1):
        return {'0_allocation': a_1}

    def kl(self, other):
        a1_dist = self._a1_distribution()
        a1_terms = a1_dist.kl(other._a1_distribution())

        return a1_terms

    @staticmethod
    @override(ActionDistribution)
    def required_model_output_shape(action_space, model_config):
        return np.prod(action_space.shape)


class TorchAutoregressiveDirichletDistributionTypeOneTestingThree(TorchDistributionWrapper):

    def a1_distribution(self):
        self.epsilon = torch.tensor(1e-7).to(self.inputs.device)
        concentration = torch.exp(self.inputs) + self.epsilon
        return torch.distributions.dirichlet.Dirichlet(
            concentration=concentration,
            validate_args=True,
        )

    @override(ActionDistribution)
    def deterministic_sample(self) -> TensorType:
        #removed depreciated warning by adding dim=1
        dist = self.a1_distribution()
        self.last_sample = nn.functional.softmax(dist.concentration, dim=1)
        return self.action_output_wrapper(self.last_sample)

    @override(ActionDistribution)
    def sample(self):
        dist = self.a1_distribution()
        self.last_sample = dist.sample()
        return self.action_output_wrapper(self.last_sample)

    @override(ActionDistribution)
    def logp(self, x):
        # Support of Dirichlet are positive real numbers. x is already
        # an array of positive numbers, but we clip to avoid zeros due to
        # numerical errors.
        dist = self.a1_distribution()
        x = torch.max(x, self.epsilon)
        x = x / torch.sum(x, dim=-1, keepdim=True)
        return dist.log_prob(x)

    @override(ActionDistribution)
    def entropy(self):
        dist = self.a1_distribution()
        return dist.entropy()

    # LEAVE THIS OUT, implementation is incorrect -> kl.kl_divergence instead of kl_divergence
    @override(ActionDistribution)
    def kl(self, other):
        dist = self.a1_distribution()
        return torch.distributions.kl.kl_divergence(dist, other.a1_distribution())#dist.kl.kl_divergence(other.a1_distribution())

    @staticmethod
    @override(ActionDistribution)
    def required_model_output_shape(action_space, model_config):
        return np.prod(action_space.shape)

    def action_output_wrapper(self, a_1):
        return {'0_allocation': a_1}

class TorchAutoregressiveDirichletDistributionTypeOneTestingFour(TorchDistributionWrapper):

    def a1_distribution(self):
        epsilon = torch.tensor(1e-7).to(self.inputs.device)
        a1_logit = self.model.a1_logits(self.inputs)
        concentration = torch.exp(a1_logit) + epsilon
        return torch.distributions.dirichlet.Dirichlet(
            concentration=concentration,
            validate_args=True,
        )

    @override(ActionDistribution)
    def deterministic_sample(self) -> TensorType:
        #removed depreciated warning by adding dim=1
        #dist = self.a1_distribution()
        epsilon = torch.tensor(1e-7).to(self.inputs.device)
        a1_logit = self.model.a1_logits(self.inputs)
        concentration = torch.exp(a1_logit) + epsilon
        self.last_sample = nn.functional.softmax(concentration, dim=1)
        return self.action_output_wrapper(self.last_sample)

    @override(ActionDistribution)
    def sample(self):
        dist = self.a1_distribution()
        self.last_sample = dist.sample()
        return self.action_output_wrapper(self.last_sample)

    @override(ActionDistribution)
    def logp(self, x):
        # Support of Dirichlet are positive real numbers. x is already
        # an array of positive numbers, but we clip to avoid zeros due to
        # numerical errors.
        epsilon = torch.tensor(1e-7).to(self.inputs.device)
        dist = self.a1_distribution()
        x = torch.max(x, epsilon)
        x = x / torch.sum(x, dim=-1, keepdim=True)
        return dist.log_prob(x)

    @override(ActionDistribution)
    def entropy(self):
        dist = self.a1_distribution()
        return dist.entropy()

    # LEAVE THIS OUT, implementation is incorrect -> kl.kl_divergence instead of kl_divergence
    @override(ActionDistribution)
    def kl(self, other):
        dist = self.a1_distribution()
        return torch.distributions.kl.kl_divergence(dist, other.a1_distribution())#dist.kl.kl_divergence(other.a1_distribution())

    @staticmethod
    @override(ActionDistribution)
    def required_model_output_shape(action_space, model_config):
        return np.prod(action_space.shape)

    def action_output_wrapper(self, a_1):
        return {'0_allocation': a_1}



class TorchAutoregressiveDirichletDistributionTypeOneTestingFive(TorchDistributionWrapper):

    def a1_distribution(self):
        epsilon = torch.tensor(1e-7).to(self.inputs.device)
        a1_logit = self.model.forward_a1(self.inputs)
        concentration = torch.exp(a1_logit) + epsilon
        return torch.distributions.dirichlet.Dirichlet(
            concentration=concentration,
            validate_args=True,
        )

    def a2_distribution(self, a1):
        epsilon = torch.tensor(1e-7).to(self.inputs.device)
        a2_logit = self.model.forward_a2(self.inputs, a1)
        concentration = torch.exp(a2_logit) + epsilon
        return torch.distributions.dirichlet.Dirichlet(
            concentration=concentration,
            validate_args=True,
        )

    @override(ActionDistribution)
    def deterministic_sample(self) -> TensorType:
        #removed depreciated warning by adding dim=1
        #dist = self.a1_distribution()
        epsilon = torch.tensor(1e-7).to(self.inputs.device)
        a1_logit = self.model.forward_a1(self.inputs)
        concentration_a1 = torch.exp(a1_logit) + epsilon
        a1 = nn.functional.softmax(concentration_a1, dim=1)

        epsilon = torch.tensor(1e-7).to(self.inputs.device)
        a2_logit = self.model.forward_a2(self.inputs, a1)
        concentration_a2 = torch.exp(a2_logit) + epsilon
        a2 = nn.functional.softmax(concentration_a2, dim=1)

        self.last_sample = torch.cat([nn.functional.softmax(concentration_a1, dim=1), nn.functional.softmax(concentration_a2, dim=1)], 1)
        return self.action_output_wrapper(a1, a2)

    @override(ActionDistribution)
    def sample(self):
        dist = self.a1_distribution()
        a1 = dist.sample()

        dist = self.a2_distribution(a1)
        a2 = dist.sample()

        self.last_sample = torch.cat(
            [a1, a2], 1)
        return self.action_output_wrapper(a1, a2)

    def normalize_dirichlet_input(self, actions):
        epsilon = torch.tensor(1e-7).to(self.inputs.device)
        actions = torch.max(actions, epsilon)
        actions = actions / torch.sum(actions, dim=-1, keepdim=True)  # normalize
        return actions

    @override(ActionDistribution)
    def logp(self, actions):
        # Support of Dirichlet are positive real numbers. x is already
        # an array of positive numbers, but we clip to avoid zeros due to
        # numerical errors.
        #Value FIX, this is a minor distortion in case all inputs are zero for the dummy run, but should not have effect on alter training
        ###

        list_actions = self.slice_action(actions)
        a1 = self.normalize_dirichlet_input(list_actions[0])
        a2 = self.normalize_dirichlet_input(list_actions[1])

        dist_a1 = self.a1_distribution()
        dist_a2 = self.a2_distribution(a1)

        return (dist_a1.log_prob(a1)+dist_a2.log_prob(a2))

    @override(ActionDistribution)
    def entropy(self):
        dist_a1 = self.a1_distribution()
        dist_a2 = self.a2_distribution(dist_a1.sample()) #TODO FIXME, we should not sample here
        return dist_a1.entropy() + dist_a2.entropy()

    # LEAVE THIS OUT, implementation is incorrect -> kl.kl_divergence instead of kl_divergence
    @override(ActionDistribution)
    def kl(self, other):
        dist_a1 = self.a1_distribution()

        a1_sample = dist_a1.sample()    #TODO FIXME, we should not sample here
        dist_a2 = self.a2_distribution(a1_sample)
        return (torch.distributions.kl.kl_divergence(dist_a2, other.a2_distribution(a1_sample)) +
                torch.distributions.kl.kl_divergence(dist_a1, other.a1_distribution()))#dist.kl.kl_divergence(other.a1_distribution())

    @staticmethod
    @override(ActionDistribution)
    def required_model_output_shape(action_space, model_config):
        return model_config.get("fcnet_hiddens")[-1] #return np.prod(action_space.shape)

    def action_output_wrapper(self, a_1, a_2):
        return {'0_allocation': a_1,
                '1_allocation': a_2}

    def slice_action(self, action):
        list_sub_actions = torch.split(action, split_size_or_sections=list(self.model.action_space_dim_dict.values()),
                                       dim=1)
        return list_sub_actions


class TorchAutoregressiveDirichletDistributionTypeOneTestingSix(TorchDistributionWrapper):

    def __init__(self, inputs, model):
        """Input is a tensor of logits. The exponential of logits is used to
        parametrize the Dirichlet distribution as all parameters need to be
        positive. An arbitrary small epsilon is added to the concentration
        parameters to be zero due to numerical error.
        See issue #4440 for more details.
        """
        epsilon = torch.tensor(1e-7).to(inputs.device)
        concentration = torch.exp(inputs) + epsilon
        #self.dist = torch.distributions.dirichlet.Dirichlet(
        #    concentration=concentration,
        #    validate_args=True,
        #)
        print("Started a New Distribution with")
        print(inputs.shape)
        super().__init__(concentration, model)

    def update_distributions(self):
        # sampled action + responsible parameters
        self.dist_parameter_a1=1 #concentration

    def a1_distribution(self):
        epsilon = torch.tensor(1e-7).to(self.inputs.device)
        a1_logit = self.model.a1_logits(self.inputs)
        concentration = torch.exp(a1_logit) + epsilon
        return torch.distributions.dirichlet.Dirichlet(
            concentration=concentration,
            validate_args=True,
        )

    @override(ActionDistribution)
    def deterministic_sample(self) -> TensorType:
        #removed depreciated warning by adding dim=1
        #dist = self.a1_distribution()

        print("called det sample")
        epsilon = torch.tensor(1e-7).to(self.inputs.device)
        a1_logit = self.model.a1_logits(self.inputs)
        concentration = torch.exp(a1_logit) + epsilon
        a1_action = nn.functional.softmax(concentration, dim=1)

        #self.last_sample = nn.functional.softmax(concentration, dim=1)
        return self.action_output_wrapper(self.last_sample)

    @override(ActionDistribution)
    def sample(self):
        print("called Sample")
        dist = self.a1_distribution()
        self.last_sample = dist.sample()
        return self.action_output_wrapper(self.last_sample)

    @override(ActionDistribution)
    def logp(self, x):
        print(x)
        list_sliced_actions = self.slice_action(x)
        x = list_sliced_actions[0]
        print("called logp")
        # Support of Dirichlet are positive real numbers. x is already
        # an array of positive numbers, but we clip to avoid zeros due to
        # numerical errors.
        epsilon = torch.tensor(1e-7).to(self.inputs.device)
        dist = self.a1_distribution()
        x = torch.max(x, epsilon)
        x = x / torch.sum(x, dim=-1, keepdim=True)
        return dist.log_prob(x)

    @override(ActionDistribution)
    def entropy(self):
        dist = self.a1_distribution()
        return dist.entropy()

    # LEAVE THIS OUT, implementation is incorrect -> kl.kl_divergence instead of kl_divergence
    @override(ActionDistribution)
    def kl(self, other):
        dist = self.a1_distribution()
        return torch.distributions.kl.kl_divergence(dist, other.a1_distribution())#dist.kl.kl_divergence(other.a1_distribution())

    @staticmethod
    @override(ActionDistribution)
    def required_model_output_shape(action_space, model_config):
        return np.prod(action_space.shape)

    def action_output_wrapper(self, a_1):
        return {'0_allocation': a_1, '0_parameter': a_1}

    def slice_action(self, action):
        print("SLICE ACTION")
        print(action.shape)
        list_sub_actions = torch.split(action, split_size_or_sections=list(self.model.action_space_dim_dict.values()),
                                       dim=1)
        return list_sub_actions


class TorchAutoregressiveDirichletDistributionTypeOneTestingSeven(TorchDistributionWrapper):

    def __init__(self, inputs, model):
        """Input is a tensor of logits. The exponential of logits is used to
        parametrize the Dirichlet distribution as all parameters need to be
        positive. An arbitrary small epsilon is added to the concentration
        parameters to be zero due to numerical error.
        See issue #4440 for more details.
        """
        self.epsilon = torch.tensor(1e-7).to(inputs.device)
        concentration = torch.exp(inputs) + self.epsilon
        #self.dist = torch.distributions.dirichlet.Dirichlet(
        #    concentration=concentration,
        #    validate_args=True,
        #)
        self.dist_a1 = None
        self.last_sample_a1 = None

        # we have to run the sampling already here to determine the entire distribution
        super().__init__(concentration, model)

        #TODO FIX ME POTENTIALLY

        a1_logit = self.model.forward_action_model(self.inputs) #a1_logits(self.inputs)
        #a1_logit = self.model.a1_logits(self.inputs)
        concentration = torch.exp(a1_logit) + self.epsilon

        self.dist_a1 = torch.distributions.dirichlet.Dirichlet(
            concentration=concentration,
            validate_args=True,
        )
        self.last_sample_a1 = self.dist_a1.sample()
        self.last_sample = torch.cat([self.last_sample_a1], 1)

        self.called_sample_already=False
        #print("Started a New Distribution with")
        #print(inputs.shape)


    def update_distribution(self, distribution_index, list_given_actions):
        #print("Update Distribution")
        if distribution_index==1:
            epsilon = torch.tensor(1e-7).to(self.inputs.device)
            a1_logit = self.model.a1_logits(self.inputs)
            concentration = torch.exp(a1_logit) + epsilon
            self.dist_a1 = torch.distributions.dirichlet.Dirichlet(
                concentration=concentration,
                validate_args=True,
            )
        else:
            raise ValueError(f"Unknown distribution index {distribution_index}")

    def a1_distribution(self):
        epsilon = torch.tensor(1e-7).to(self.inputs.device)
        a1_logit = self.model.a1_logits(self.inputs)
        concentration = torch.exp(a1_logit) + epsilon
        return torch.distributions.dirichlet.Dirichlet(
            concentration=concentration,
            validate_args=True,
        )

    @override(ActionDistribution)
    def deterministic_sample(self) -> TensorType:
        #removed depreciated warning by adding dim=1
        #dist = self.a1_distribution()
        #for dummy purposes
        #print("Det sample")
        #self.update_distribution(distribution_index=1, list_given_actions=[])

        epsilon = torch.tensor(1e-7).to(self.inputs.device)
        a1_logit = self.model.a1_logits(self.inputs)
        concentration = torch.exp(a1_logit) + epsilon
        self.last_sample_a1 = nn.functional.softmax(concentration, dim=1)
        self.last_sample = torch.cat([self.last_sample_a1], 1)
        return self.action_output_wrapper(self.last_sample_a1)

    def sample_initial(self):
        #print("Call Sample")
        self.update_distribution(distribution_index=1, list_given_actions=[])
        # dist = self.a1_distribution()
        self.last_sample_a1 = self.dist_a1.sample()

        self.last_sample = torch.cat([self.last_sample_a1], 1)
        return self.action_output_wrapper(self.last_sample_a1)

    @override(ActionDistribution)
    def sample(self):
        #print("Call Sample")
        #self.update_distribution(distribution_index=1, list_given_actions=[])
        #dist = self.a1_distribution()
        #self.last_sample_a1 = self.dist_a1.sample()
        assert not self.called_sample_already
        self.called_sample_already=True
        #self.last_sample = torch.cat([self.last_sample_a1], 1)
        return self.action_output_wrapper(self.last_sample_a1)

    @override(ActionDistribution)
    def logp(self, x):
        # Support of Dirichlet are positive real numbers. x is already
        # an array of positive numbers, but we clip to avoid zeros due to
        # numerical errors.
        #if we call logp we should have already created all distributions
        #print("Call logp")
        #print(x)
        if self.dist_a1 is None:
            raise ValueError(f'dist_a1 should not be None')

        epsilon = torch.tensor(1e-7).to(self.inputs.device)
        #dist = self.a1_distribution()
        x = torch.max(x, epsilon)
        x = x / torch.sum(x, dim=-1, keepdim=True)
        return self.dist_a1.log_prob(x)

    @override(ActionDistribution)
    def entropy(self):
        #dist = self.a1_distribution()
        return self.dist_a1.entropy()

    # LEAVE THIS OUT, implementation is incorrect -> kl.kl_divergence instead of kl_divergence
    @override(ActionDistribution)
    def kl(self, other):
        #dist = self.a1_distribution()
        return torch.distributions.kl.kl_divergence(self.dist_a1, other.dist_a1)#dist.kl.kl_divergence(other.a1_distribution())

    @staticmethod
    @override(ActionDistribution)
    def required_model_output_shape(action_space, model_config):
        return np.prod(action_space.shape)

    def action_output_wrapper(self, a_1):
        return {'0_allocation': a_1}

    def slice_action(self, action):
        #print("SLICE ACTION")
        #print(action.shape)
        list_sub_actions = torch.split(action, split_size_or_sections=list(self.model.action_space_dim_dict.values()),
                                       dim=1)
        return list_sub_actions



class TorchAutoregressiveDirichletDistributionTypeOneTestingEight(TorchDistributionWrapper):

    def __init__(self, inputs, model):
        """Input is a tensor of logits. The exponential of logits is used to
        parametrize the Dirichlet distribution as all parameters need to be
        positive. An arbitrary small epsilon is added to the concentration
        parameters to be zero due to numerical error.
        See issue #4440 for more details.
        """
        self.epsilon = torch.tensor(1e-7).to(inputs.device)
        concentration = torch.exp(inputs) + self.epsilon
        #self.dist = torch.distributions.dirichlet.Dirichlet(
        #    concentration=concentration,
        #    validate_args=True,
        #)
        self.dist_a1 = None
        self.last_sample_a1 = None
        self.dist_a2 = None
        self.last_sample_a2 = None

        # we have to run the sampling already here to determine the entire distribution
        super().__init__(concentration, model)

        #TODO FIX ME POTENTIALLY

        a1_logit = self.model.forward_action_model_a1(self.inputs) #a1_logits(self.inputs)
        #a1_logit = self.model.a1_logits(self.inputs)
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
        #print("Started a New Distribution with")
        #print(inputs.shape)


    def update_distribution(self, distribution_index, list_given_actions):
        #print("Update Distribution")
        if distribution_index==1:
            epsilon = torch.tensor(1e-7).to(self.inputs.device)
            a1_logit = self.model.a1_logits(self.inputs)
            concentration = torch.exp(a1_logit) + epsilon
            self.dist_a1 = torch.distributions.dirichlet.Dirichlet(
                concentration=concentration,
                validate_args=True,
            )
        else:
            raise ValueError(f"Unknown distribution index {distribution_index}")

    def a1_distribution(self):
        epsilon = torch.tensor(1e-7).to(self.inputs.device)
        a1_logit = self.model.a1_logits(self.inputs)
        concentration = torch.exp(a1_logit) + epsilon
        return torch.distributions.dirichlet.Dirichlet(
            concentration=concentration,
            validate_args=True,
        )

    @override(ActionDistribution)
    def deterministic_sample(self) -> TensorType:
        #removed depreciated warning by adding dim=1
        #dist = self.a1_distribution()
        #for dummy purposes
        #print("Det sample")
        #self.update_distribution(distribution_index=1, list_given_actions=[])

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
        #print("Call Sample")
        #self.update_distribution(distribution_index=1, list_given_actions=[])
        #dist = self.a1_distribution()
        #self.last_sample_a1 = self.dist_a1.sample()
        assert not self.called_sample_already
        self.called_sample_already=True
        #self.last_sample = torch.cat([self.last_sample_a1], 1)
        return self.action_output_wrapper(self.last_sample)

    @override(ActionDistribution)
    def logp(self, action):
        # Support of Dirichlet are positive real numbers. x is already
        # an array of positive numbers, but we clip to avoid zeros due to
        # numerical errors.
        #if we call logp we should have already created all distributions
        #print("Call logp")
        #print(x)
        if self.dist_a1 is None or self.dist_a2 is None:
            raise ValueError(f'dist_a1 or dist_a2 should not be None')

        list_actions = self.slice_action(action)
        action_a1 = list_actions[0]
        action_a2 = list_actions[1]
        #print(list_actions)
        epsilon_a1 = torch.tensor(1e-7).to(self.inputs.device)
        #dist = self.a1_distribution()
        action_a1 = torch.max(action_a1, epsilon_a1)
        action_a1 = action_a1 / torch.sum(action_a1, dim=-1, keepdim=True)

        epsilon_a2 = torch.tensor(1e-7).to(self.inputs.device)
        action_a2 = torch.max(action_a2, epsilon_a2)
        action_a2 = action_a2 / torch.sum(action_a2, dim=-1, keepdim=True)

        return self.dist_a1.log_prob(action_a1)+self.dist_a2.log_prob(action_a2)

    @override(ActionDistribution)
    def entropy(self):
        #dist = self.a1_distribution()
        return self.dist_a1.entropy()+self.dist_a2.entropy()

    # LEAVE THIS OUT, implementation is incorrect -> kl.kl_divergence instead of kl_divergence
    @override(ActionDistribution)
    def kl(self, other):
        #dist = self.a1_distribution()
        kl_dist1 = torch.distributions.kl.kl_divergence(self.dist_a1, other.dist_a1)
        kl_dist2 = torch.distributions.kl.kl_divergence(self.dist_a2, other.dist_a2)
        return kl_dist1+ kl_dist2#dist.kl.kl_divergence(other.a1_distribution())

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return model_config.get("fcnet_hiddens")[
            -1]  # self.model.hidden_output_size#16  # controls model output feature vector size

    def action_output_wrapper(self, action):
        list_actions = self.slice_action(action)
        return {'0_allocation': list_actions[0], '1_allocation': list_actions[1]}

    def slice_action(self, action):
        #print("SLICE ACTION")
        #print(action.shape)
        #print(list(self.model.action_space_dim_dict.values()))
        list_sub_actions = torch.split(action, split_size_or_sections=list(self.model.action_space_dim_dict.values()),
                                       dim=1)
        return list_sub_actions



class TorchAutoregressiveDirichletDistributionTypeOneTestingNine(TorchDistributionWrapper):

    def __init__(self, inputs, model):
        """Input is a tensor of logits. The exponential of logits is used to
        parametrize the Dirichlet distribution as all parameters need to be
        positive. An arbitrary small epsilon is added to the concentration
        parameters to be zero due to numerical error.
        See issue #4440 for more details.
        """
        self.epsilon = torch.tensor(1e-7).to(inputs.device)
        concentration = torch.exp(inputs) + self.epsilon
        #self.dist = torch.distributions.dirichlet.Dirichlet(
        #    concentration=concentration,
        #    validate_args=True,
        #)
        self.dist_a1 = None
        self.last_sample_a1 = None
        self.dist_a2 = None
        self.last_sample_a2 = None

        # we have to run the sampling already here to determine the entire distribution
        super().__init__(concentration, model)

        #TODO FIX ME POTENTIALLY

        a1_logit = self.model.forward_action_model_a1(self.inputs) #a1_logits(self.inputs)
        #a1_logit = self.model.a1_logits(self.inputs)
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
        #removed depreciated warning by adding dim=1
        #dist = self.a1_distribution()
        #for dummy purposes
        #print("Det sample")
        #self.update_distribution(distribution_index=1, list_given_actions=[])

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
        #print("Call Sample")
        #self.update_distribution(distribution_index=1, list_given_actions=[])
        #dist = self.a1_distribution()
        #self.last_sample_a1 = self.dist_a1.sample()
        assert not self.called_sample_already
        self.called_sample_already=True
        #self.last_sample = torch.cat([self.last_sample_a1], 1)
        return self.action_output_wrapper(self.last_sample)

    @override(ActionDistribution)
    def logp(self, action):
        # Support of Dirichlet are positive real numbers. x is already
        # an array of positive numbers, but we clip to avoid zeros due to
        # numerical errors.
        #if we call logp we should have already created all distributions
        #print("Call logp")
        #print(x)
        if self.dist_a1 is None or self.dist_a2 is None:
            raise ValueError(f'dist_a1 or dist_a2 should not be None')

        list_actions = self.slice_action(action)
        action_a1 = list_actions[0]
        action_a2 = list_actions[1]
        #print(list_actions)
        epsilon_a1 = torch.tensor(1e-7).to(self.inputs.device)
        #dist = self.a1_distribution()
        action_a1 = torch.max(action_a1, epsilon_a1)
        action_a1 = action_a1 / torch.sum(action_a1, dim=-1, keepdim=True)

        epsilon_a2 = torch.tensor(1e-7).to(self.inputs.device)
        action_a2 = torch.max(action_a2, epsilon_a2)
        action_a2 = action_a2 / torch.sum(action_a2, dim=-1, keepdim=True)

        return self.dist_a1.log_prob(action_a1)+self.dist_a2.log_prob(action_a2)

    @override(ActionDistribution)
    def entropy(self):
        #dist = self.a1_distribution()
        return self.dist_a1.entropy()+self.dist_a2.entropy()

    # LEAVE THIS OUT, implementation is incorrect -> kl.kl_divergence instead of kl_divergence
    @override(ActionDistribution)
    def kl(self, other):
        #dist = self.a1_distribution()
        kl_dist1 = torch.distributions.kl.kl_divergence(self.dist_a1, other.dist_a1)
        kl_dist2 = torch.distributions.kl.kl_divergence(self.dist_a2, other.dist_a2)
        return kl_dist1+ kl_dist2#dist.kl.kl_divergence(other.a1_distribution())

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return model_config.get("fcnet_hiddens")[
            -1]  # self.model.hidden_output_size#16  # controls model output feature vector size

    def action_output_wrapper(self, action):
        list_actions = self.slice_action(action)
        return {'0_allocation': list_actions[0], '1_allocation': list_actions[1]}

    def slice_action(self, action):
        #print("SLICE ACTION")
        #print(action.shape)
        #print(list(self.model.action_space_dim_dict.values()))
        list_sub_actions = torch.split(action, split_size_or_sections=list(self.model.action_space_dim_dict.values()),
                                       dim=1)
        return list_sub_actions
