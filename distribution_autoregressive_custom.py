from ray.rllib.models.tf.tf_action_dist import Categorical, ActionDistribution
from ray.rllib.models.torch.torch_action_dist import TorchCategorical, \
    TorchDistributionWrapper
from ray.rllib.utils.framework import try_import_tf, try_import_torch

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()


class BinaryAutoregressiveDistribution(ActionDistribution):
    """Action distribution P(a1, a2) = P(a1) * P(a2 | a1)"""

    def deterministic_sample(self):
        # First, sample a1.
        a1_dist = self._a1_distribution()
        a1 = a1_dist.deterministic_sample()

        # Sample a2 conditioned on a1.
        a2_dist = self._a2_distribution(a1)
        a2 = a2_dist.deterministic_sample()
        self._action_logp = a1_dist.logp(a1) + a2_dist.logp(a2)

        # Return the action tuple.
        return (a1, a2)

    def sample(self):
        # First, sample a1.
        a1_dist = self._a1_distribution()
        a1 = a1_dist.sample()

        # Sample a2 conditioned on a1.
        a2_dist = self._a2_distribution(a1)
        a2 = a2_dist.sample()
        self._action_logp = a1_dist.logp(a1) + a2_dist.logp(a2)

        # Return the action tuple.
        return (a1, a2)

    def logp(self, actions):
        a1, a2 = actions[:, 0], actions[:, 1]
        a1_vec = tf.expand_dims(tf.cast(a1, tf.float32), 1)
        a1_logits, a2_logits = self.model.action_model([self.inputs, a1_vec])
        return (
            Categorical(a1_logits).logp(a1) + Categorical(a2_logits).logp(a2))

    def sampled_action_logp(self):
        return tf.exp(self._action_logp)

    def entropy(self):
        a1_dist = self._a1_distribution()
        a2_dist = self._a2_distribution(a1_dist.sample())
        return a1_dist.entropy() + a2_dist.entropy()

    def kl(self, other):
        a1_dist = self._a1_distribution()
        a1_terms = a1_dist.kl(other._a1_distribution())

        a1 = a1_dist.sample()
        a2_terms = self._a2_distribution(a1).kl(other._a2_distribution(a1))
        return a1_terms + a2_terms

    def _a1_distribution(self):
        BATCH = tf.shape(self.inputs)[0]
        a1_logits, _ = self.model.action_model(
            [self.inputs, tf.zeros((BATCH, 1))])
        a1_dist = Categorical(a1_logits)
        return a1_dist

    def _a2_distribution(self, a1):
        a1_vec = tf.expand_dims(tf.cast(a1, tf.float32), 1)
        _, a2_logits = self.model.action_model([self.inputs, a1_vec])
        a2_dist = Categorical(a2_logits)
        return a2_dist

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return 16  # controls model output feature vector size


class TorchBinaryAutoregressiveDistribution(TorchDistributionWrapper):
    """Action distribution P(a1, a2) = P(a1) * P(a2 | a1)"""

    def deterministic_sample(self):
        # First, sample a1.
        a1_dist = self._a1_distribution()
        a1 = a1_dist.deterministic_sample()

        # Sample a2 conditioned on a1.
        a2_dist = self._a2_distribution(a1)
        a2 = a2_dist.deterministic_sample()
        self._action_logp = a1_dist.logp(a1) + a2_dist.logp(a2)

        # Return the action tuple.
        #return (a1, a2)
        #print(self.action_output_wrapper(a1, a2))
        return self.action_output_wrapper(a1, a2)

    def sample(self):
        # First, sample a1.
        a1_dist = self._a1_distribution()
        a1 = a1_dist.sample()

        # Sample a2 conditioned on a1.
        a2_dist = self._a2_distribution(a1)
        a2 = a2_dist.sample()
        self._action_logp = a1_dist.logp(a1) + a2_dist.logp(a2)

        # Return the action tuple.
        #return (a1, a2)
        #print(self.action_output_wrapper(a1, a2))
        return self.action_output_wrapper(a1, a2)

    def logp(self, actions):
        a1, a2 = actions[:, 0], actions[:, 1]
        a1_vec = torch.unsqueeze(a1.float(), 1)
        a1_logits, a2_logits = self.model.action_module(self.inputs, a1_vec)
        return (TorchCategorical(a1_logits).logp(a1) +
                TorchCategorical(a2_logits).logp(a2))

    def sampled_action_logp(self):
        return torch.exp(self._action_logp)

    def entropy(self):
        a1_dist = self._a1_distribution()
        a2_dist = self._a2_distribution(a1_dist.sample())
        return a1_dist.entropy() + a2_dist.entropy()

    def kl(self, other):
        a1_dist = self._a1_distribution()
        a1_terms = a1_dist.kl(other._a1_distribution())

        a1 = a1_dist.sample()
        a2_terms = self._a2_distribution(a1).kl(other._a2_distribution(a1))
        return a1_terms + a2_terms

    def _a1_distribution(self):
        BATCH = self.inputs.shape[0]
        zeros = torch.zeros((BATCH, 1)).to(self.inputs.device)
        a1_logits, _ = self.model.action_module(self.inputs, zeros)
        a1_dist = TorchCategorical(a1_logits)
        return a1_dist

    def _a2_distribution(self, a1):
        a1_vec = torch.unsqueeze(a1.float(), 1)
        _, a2_logits = self.model.action_module(self.inputs, a1_vec)
        a2_dist = TorchCategorical(a2_logits)
        return a2_dist

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        print("CALLING THE REQU MODEL")
        print("-------------------------")
        print("-------------------------")
        print("-------------------------")
        print("-------------------------")
        return 16  # controls model output feature vector size

    def action_output_wrapper(self, a_1, a_2):
        return {'a_1': a_1,
                'a_2': a_2}


from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper,TorchMultiActionDistribution,TorchBeta,TorchCategorical
from ray.rllib.utils.spaces.space_utils import get_base_struct_from_space
import tree
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

class My_betadist(TorchMultiActionDistribution,TorchDistributionWrapper):
    def __init__(self, inputs, model, *, child_distributions, input_lens,
                 action_space):
        # super().__init__(inputs, model, child_distributions, input_lens,action_space)
        #child_distributions = [TorchBeta,TorchBeta,TorchCategorical]
        child_distributions = [TorchCategorical, TorchCategorical]
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.from_numpy(inputs)
            if isinstance(model, TorchModelV2):
                inputs = inputs.to(next(model.parameters()).device)
        # print(inputs,'aaa')
        #print("MYBETADIST")

        TorchDistributionWrapper.__init__(self,inputs, model)
        self.action_space_struct = get_base_struct_from_space(action_space)
        self.input_lens = tree.flatten(input_lens)
        flat_child_distributions = tree.flatten(child_distributions)
        split_inputs = torch.split(inputs, self.input_lens, dim=1)
        self.flat_child_distributions = tree.map_structure(
            lambda dist, input_: dist(input_, model), flat_child_distributions,
            list(split_inputs))
        print(self.flat_child_distributions)

    #@override(ActionDistribution)
    def sample(self):
        print("SAMPLE")
        child_distributions = tree.unflatten_as(self.action_space_struct,
                                              self.flat_child_distributions)
        print(type(tree.map_structure(lambda s: s.sample(), child_distributions)))
        print(tree.map_structure(lambda s: s.sample(), child_distributions))
        return tree.map_structure(lambda s: s.sample(), child_distributions)

    #@override(ActionDistribution)
    def deterministic_sample(self):
        child_distributions = tree.unflatten_as(self.action_space_struct,
                                              self.flat_child_distributions)
        return tree.map_structure(lambda s: s.deterministic_sample(),
                                child_distributions)

from dirichlet_custom import TorchDirichlet_Custom

class TorchAutoregressiveDirichletDistribution(TorchDistributionWrapper):
    """Action distribution P(a1, a2) = P(a1) * P(a2 | a1)"""

    def deterministic_sample(self):
        # First, sample a1.
        a1_dist = self._a1_distribution()
        a1 = a1_dist.deterministic_sample()

        # Sample a2 conditioned on a1.
        a2_dist = self._a2_distribution(a1)
        a2 = a2_dist.deterministic_sample()
        self._action_logp = a1_dist.logp(a1) + a2_dist.logp(a2)

        # Return the action tuple.
        #return (a1, a2)
        #print(self.action_output_wrapper(a1, a2))
        return self.action_output_wrapper(a1, a2)

    def sample(self):
        # First, sample a1.
        a1_dist = self._a1_distribution()
        a1 = a1_dist.sample()

        # Sample a2 conditioned on a1.
        a2_dist = self._a2_distribution(a1)
        a2 = a2_dist.sample()
        self._action_logp = a1_dist.logp(a1) + a2_dist.logp(a2)

        # Return the action tuple.
        #return (a1, a2)
        #print(self.action_output_wrapper(a1, a2))
        return self.action_output_wrapper(a1, a2)

    def logp(self, actions):
        #print("LOGP")
        #print(actions)
        #print(self.model.action_space_dim_dict)
        #print("IMPORTANT")
        #a1, a2 = actions[:, 0], actions[:, 1]

        list_actions = self.slice_action(actions)
        a1 = list_actions[0]
        a2 = list_actions[1]

        #a1_vec = torch.unsqueeze(a1.float(), 1)
        #print("A1VEC")
        #print(a1_vec.shape)
        #print(self.inputs.shape)
        #print(a1.shape)
        #print(a2.shape)
        #print("---")
        #a1_logits, a2_logits = self.model.action_module(self.inputs, a1_vec)
        a1_logits, a2_logits = self.model.action_module(self.inputs, a1)
        #by doing this we safe one calculation
        return (TorchDirichlet_Custom(a1_logits, self.model).logp(a1) +
                TorchDirichlet_Custom(a2_logits, self.model).logp(a2))

    def sampled_action_logp(self):
        return torch.exp(self._action_logp)

    def entropy(self):
        a1_dist = self._a1_distribution()
        a2_dist = self._a2_distribution(a1_dist.sample())
        return a1_dist.entropy() + a2_dist.entropy()

    def kl(self, other):
        a1_dist = self._a1_distribution()
        a1_terms = a1_dist.kl(other._a1_distribution())

        a1 = a1_dist.sample()
        a2_terms = self._a2_distribution(a1).kl(other._a2_distribution(a1))
        return a1_terms + a2_terms

    def _a1_distribution(self):
        BATCH = self.inputs.shape[0]
        #dummies for a2
        zeros = torch.zeros((BATCH, self.model.action_space_dim_dict.get("a_2"))).to(self.inputs.device)
        a1_logits, _ = self.model.action_module(self.inputs, zeros)
        a1_dist = TorchDirichlet_Custom(a1_logits, self.model)
        return a1_dist

    def _a2_distribution(self, a1):
        #a1_vec = torch.unsqueeze(a1.float(), 1) # probably we do this because we had a single value in the example
        #_, a2_logits = self.model.action_module(self.inputs, a1_vec)
        _, a2_logits = self.model.action_module(self.inputs, a1)
        a2_dist = TorchDirichlet_Custom(a2_logits, self.model)
        return a2_dist

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        print("CALLING THE REQU MODEL")
        print("-------------------------")
        print("-------------------------")
        print("-------------------------")
        print("-------------------------")
        return 16  # controls model output feature vector size

    def action_output_wrapper(self, a_1, a_2):
        return {'a_1': a_1,
                'a_2': a_2}

    def slice_action(self, action):
        list_sub_actions = torch.split(action, split_size_or_sections=list(self.model.action_space_dim_dict.values()),
                                       dim=1)
        return list_sub_actions

class TorchAutoregressiveDirichletDistributionV2(TorchDistributionWrapper):
    """Action distribution P(a1, a2) = P(a1) * P(a2 | a1)"""

    def deterministic_sample(self):
        # First, sample a1.
        a1_dist = self._a1_distribution()
        a1 = a1_dist.deterministic_sample()

        # Sample a2 conditioned on a1.
        a2_dist = self._a2_distribution(a1)
        a2 = a2_dist.deterministic_sample()
        self._action_logp = a1_dist.logp(a1) + a2_dist.logp(a2)

        # Return the action tuple.
        #return (a1, a2)
        #print(self.action_output_wrapper(a1, a2))
        return self.action_output_wrapper(a1, a2)

    def sample(self):
        # First, sample a1.
        a1_dist = self._a1_distribution()
        a1 = a1_dist.sample()

        # Sample a2 conditioned on a1.
        a2_dist = self._a2_distribution(a1)
        a2 = a2_dist.sample()
        self._action_logp = a1_dist.logp(a1) + a2_dist.logp(a2)

        # Return the action tuple.
        #return (a1, a2)
        #print(self.action_output_wrapper(a1, a2))
        return self.action_output_wrapper(a1, a2)

    def logp(self, actions):
        #print("LOGP")
        #print(actions)
        #print(self.model.action_space_dim_dict)
        #print("IMPORTANT")
        #a1, a2 = actions[:, 0], actions[:, 1]

        list_actions = self.slice_action(actions)
        a1 = list_actions[0]
        a2 = list_actions[1]

        #a1_vec = torch.unsqueeze(a1.float(), 1)
        #print("A1VEC")
        #print(a1_vec.shape)
        #print(self.inputs.shape)
        #print(a1.shape)
        #print(a2.shape)
        #print("---")
        #a1_logits, a2_logits = self.model.action_module(self.inputs, a1_vec)
        #a1_logits, a2_logits = self.model.action_module(self.inputs, a1)
        #a1_logits, a2_logits = self.model.forward_action_model(self.inputs, a1)
        a1_logits, a2_logits = self.model.forward_action_model(self.inputs, [a1, None]) #last entry is not important
        #by doing this we safe one calculation
        return (TorchDirichlet_Custom(a1_logits, self.model).logp(a1) +
                TorchDirichlet_Custom(a2_logits, self.model).logp(a2))

    def sampled_action_logp(self):
        return torch.exp(self._action_logp)

    def entropy(self):
        a1_dist = self._a1_distribution()
        a2_dist = self._a2_distribution(a1_dist.sample())
        return a1_dist.entropy() + a2_dist.entropy()

    def kl(self, other):
        a1_dist = self._a1_distribution()
        a1_terms = a1_dist.kl(other._a1_distribution())

        a1 = a1_dist.sample()
        a2_terms = self._a2_distribution(a1).kl(other._a2_distribution(a1))
        return a1_terms + a2_terms

    def _a_distribution(self, action_idx, list_action_input):
        """
        :param action_idx: uses zero indexing
        :param list_action_input:
        :return:
        """
        assert action_idx==len(list_action_input)
        BATCH_SIZE = self.inputs.shape[0]
        tmp_list_input_actions = []

        #building inputs
        for ctx, (key, value) in enumerate(self.model.action_space_dim_dict.items()):
            if action_idx>ctx:
                tmp_list_input_actions.append(list_action_input[ctx])
            else:
                tmp_list_input_actions.append(torch.zeros((BATCH_SIZE, value)).to(self.inputs.device))

        list_logits = self.model.forward_action_model(self.inputs, tmp_list_input_actions)
        a_dist = TorchDirichlet_Custom(list_logits[action_idx], self.model)
        return a_dist

    def _a1_distribution(self):
        #BATCH = self.inputs.shape[0]
        #dummies for a2
        #zeros = torch.zeros((BATCH, self.model.action_space_dim_dict.get("a_2"))).to(self.inputs.device)
        #a1_logits, _ = self.model.forward_action_model(self.inputs, zeros)
        #a1_logits, _ = self.model.action_module(self.inputs, zeros)
        #a1_dist = TorchDirichlet_Custom(a1_logits, self.model)
        #return a1_dist
        return self._a_distribution(action_idx=0, list_action_input=[])

    def _a2_distribution(self, a1):
        #a1_vec = torch.unsqueeze(a1.float(), 1) # probably we do this because we had a single value in the example
        #_, a2_logits = self.model.action_module(self.inputs, a1_vec)
        #_, a2_logits = self.model.action_module(self.inputs, a1)
        #_, a2_logits = self.model.forward_action_model(self.inputs, a1)
        #a2_dist = TorchDirichlet_Custom(a2_logits, self.model)
        #return a2_dist
        return self._a_distribution(action_idx=1, list_action_input=[a1])

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return model_config.get("fcnet_hiddens")[-1] #self.model.hidden_output_size#16  # controls model output feature vector size

    def action_output_wrapper(self, a_1, a_2):
        #return {'a_1': a_1,
        #        'a_2': a_2}
        return {'0_allocation': a_1,
                '1_allocation': a_2}

    def slice_action(self, action):
        list_sub_actions = torch.split(action, split_size_or_sections=list(self.model.action_space_dim_dict.values()),
                                       dim=1)
        return list_sub_actions