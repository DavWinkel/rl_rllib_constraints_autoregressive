import logging
import numpy as np
import gym

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, AppendBiasLayer, normc_initializer
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict
from ray.rllib.utils.spaces.simplex import Simplex
from ray.rllib.utils.torch_utils import FLOAT_MIN

import re

from moment_model import MomentModel
from moment_model_transformer import MomentTransformerModel

torch, nn = try_import_torch()

logger = logging.getLogger(__name__)


class TorchCustomAutoregressiveModelS4(TorchModelV2, nn.Module):
    """Custom autoregressive fully connected network."""

    def __init__(
            self,
            obs_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            num_outputs: int,
            model_config: ModelConfigDict,
            name: str,
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        orig_action_space = getattr(action_space, "original_space", action_space)
        orig_space = getattr(obs_space, "original_space", obs_space)

        self.obs_space_dim_dict = self.generate_dict_gym_space_meta_coordinates(gym_space=orig_space)
        self.action_space_dim_dict = self.generate_dict_gym_space_meta_coordinates(gym_space=orig_action_space)

        hiddens = list(model_config.get("fcnet_hiddens", [])) + list(
            model_config.get("post_fcnet_hiddens", [])
        )

        self.hidden_output_size = hiddens[-1]

        activation = model_config.get("fcnet_activation")

        self.vf_share_layers = model_config.get("vf_share_layers")

        substring_action_mask = "action_mask"
        substring_action_head = "head_factor"

        self.obs_size = 0
        for key, value in self.obs_space_dim_dict.items():
            if not re.search(substring_action_mask, key) and not re.search(substring_action_head, key):
                self.obs_size += value

        #self.dict_action_masks = None
        self.current_action_masks = None

        layers = []
        prev_layer_size = self.obs_size  # int(np.product(obs_space.shape))
        self._logits = None

        # Create layers 0 to second-last.
        for size in hiddens[:-1]:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=size,
                    initializer=normc_initializer(1.0),
                    activation_fn=activation,
                )
            )
            prev_layer_size = size

        # generate last layer for context layer
        if len(hiddens) > 0:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=hiddens[-1],
                    initializer=normc_initializer(1.0),
                    activation_fn=activation,
                )
            )
            prev_layer_size = hiddens[-1]


        self._hidden_layers = nn.Sequential(*layers)


        self._value_branch = SlimFC(
            in_size=prev_layer_size,
            out_size=1,
            initializer=normc_initializer(0.01),
            activation_fn=None,
        )
        # Holds the current "base" output (before logits layer).
        self._features = None
        # Holds the last input, in case value branch is separate.
        self._last_flat_in = None

        #Hiddens for the autoregressive branch
        if "custom_model_config" in model_config:
            hiddens_action_branch = list(model_config.get("custom_model_config").get("fcnet_hiddens_autoreg_branches", []))
        else:
            raise ValueError("Please provide custom_model_config.fcnet_hiddens_autoreg_branches")


        ## a1
        tmp_layers_action_branch_a1=[]
        pre_layer_size_action_branch_a1 = self.hidden_output_size # prev_layer_size

        #hiddens for the actions:
        for size in hiddens_action_branch[:-1]:
            tmp_layers_action_branch_a1.append(
                SlimFC(
                    in_size=pre_layer_size_action_branch_a1,
                    out_size=size,
                    initializer=normc_initializer(1.0),
                    activation_fn=activation,
                )
            )
            pre_layer_size_action_branch_a1 = size

        # add final logit layer
        tmp_layers_action_branch_a1.append(
            SlimFC(
                in_size=pre_layer_size_action_branch_a1,  # TODO fixme
                out_size=self.action_space_dim_dict.get("0_allocation"),  # self.num_outputs_action_distr, #this was 2 before
                activation_fn=None,
                initializer=normc_initializer(0.01),
            )
        )

        self.a1_logit_layers = nn.Sequential(*tmp_layers_action_branch_a1)
        ##End a1

        ## a2
        tmp_layers_action_branch_a2 = []
        pre_layer_size_action_branch_a2 = self.hidden_output_size + self.action_space_dim_dict.get("0_allocation") #prev_layer_size
        # hiddens for the actions:
        for size in hiddens_action_branch[:-1]:
            tmp_layers_action_branch_a2.append(
                SlimFC(
                    in_size=pre_layer_size_action_branch_a2,
                    out_size=size,
                    initializer=normc_initializer(1.0),
                    activation_fn=activation,
                )
            )
            pre_layer_size_action_branch_a2 = size

        # add final logit layer
        tmp_layers_action_branch_a2.append(
            SlimFC(
                in_size=pre_layer_size_action_branch_a2,  # TODO fixme
                out_size=self.action_space_dim_dict.get("1_allocation"),
                activation_fn=None,
                initializer=normc_initializer(0.01),
            )
        )
        self.a2_logit_layers = nn.Sequential(*tmp_layers_action_branch_a2)
        ## End a2

        ## a3
        tmp_layers_action_branch_a3 = []
        pre_layer_size_action_branch_a3 = self.hidden_output_size + self.action_space_dim_dict.get(
            "0_allocation") + self.action_space_dim_dict.get(
            "1_allocation")  # prev_layer_size
        # hiddens for the actions:
        for size in hiddens_action_branch[:-1]:
            tmp_layers_action_branch_a3.append(
                SlimFC(
                    in_size=pre_layer_size_action_branch_a3,
                    out_size=size,
                    initializer=normc_initializer(1.0),
                    activation_fn=activation,
                )
            )
            pre_layer_size_action_branch_a3 = size

        # add final logit layer
        tmp_layers_action_branch_a3.append(
            SlimFC(
                in_size=pre_layer_size_action_branch_a3,  # TODO fixme
                out_size=self.action_space_dim_dict.get("2_allocation"),
                activation_fn=None,
                initializer=normc_initializer(0.01),
            )
        )
        self.a3_logit_layers = nn.Sequential(*tmp_layers_action_branch_a3)
        ## End a3

        ## a4
        tmp_layers_action_branch_a4 = []
        pre_layer_size_action_branch_a4 = self.hidden_output_size + self.action_space_dim_dict.get(
            "0_allocation") + self.action_space_dim_dict.get(
            "1_allocation") + self.action_space_dim_dict.get(
            "2_allocation")  # prev_layer_size
        # hiddens for the actions:
        for size in hiddens_action_branch[:-1]:
            tmp_layers_action_branch_a4.append(
                SlimFC(
                    in_size=pre_layer_size_action_branch_a4,
                    out_size=size,
                    initializer=normc_initializer(1.0),
                    activation_fn=activation,
                )
            )
            pre_layer_size_action_branch_a4 = size

        # add final logit layer
        tmp_layers_action_branch_a4.append(
            SlimFC(
                in_size=pre_layer_size_action_branch_a4,  # TODO fixme
                out_size=self.action_space_dim_dict.get("2_allocation"),
                activation_fn=None,
                initializer=normc_initializer(0.01),
            )
        )
        self.a4_logit_layers = nn.Sequential(*tmp_layers_action_branch_a4)
        ## End a4


        #Introduce risk model
        if "custom_model_config" in model_config and "config_moment_model" in model_config.get("custom_model_config"):

            number_input_assets = self.action_space_dim_dict.get("0_allocation")
            moment_model_config = model_config.get("custom_model_config").get("config_moment_model")
            self.moment_model_attention_dim = model_config.get("custom_model_config").get("config_moment_model").get(
                "attention_dim")

            self.use_moment_attention = model_config.get("custom_model_config").get("config_moment_model").get(
                "use_moment_attention")

            if self.use_moment_attention:
                self.moment_submodel = MomentTransformerModel(config_attention_model=moment_model_config,
                                                              in_space=obs_space, number_input_assets=number_input_assets) #action_space=action_space)
            else:
                self.moment_submodel = MomentModel(config_moment_model=moment_model_config,
                                                   in_space=obs_space, action_space=action_space)
        else:
            print("No moment model currently used")

    def store_input_action_masks(self, input_dict):
        substring_action_mask = "action_mask"
        dict_action_mask = {}
        for key in self.obs_space_dim_dict.keys():
            if re.search(substring_action_mask, key):
                mask_idx = key.split("_")[0]
                tmp_action_mask = input_dict["obs"].get(key)
                dict_action_mask[mask_idx] = tmp_action_mask

        return dict_action_mask


    def forward_action_model_a1(self, ctx_input):

        a1_logit = self.a1_logit_layers(ctx_input)

        inf_mask = torch.clamp(torch.log(self.current_action_masks.get('0')), min=FLOAT_MIN)

        tmp_masked_logit = a1_logit + inf_mask

        return tmp_masked_logit

    def forward_action_model_a2(self, ctx_input, a1_action):

        a2_cat_input = torch.cat([ctx_input, a1_action], 1)

        a2_logit = self.a2_logit_layers(a2_cat_input)

        inf_mask = torch.clamp(torch.log(self.current_action_masks.get('1')), min=FLOAT_MIN)

        tmp_masked_logit = a2_logit + inf_mask

        return tmp_masked_logit


    def forward_action_model_a3(self, ctx_input, a1_action, a2_action):

        a3_cat_input = torch.cat([ctx_input, a1_action, a2_action], 1)

        a3_logit = self.a3_logit_layers(a3_cat_input)

        inf_mask = torch.clamp(torch.log(self.current_action_masks.get('2')), min=FLOAT_MIN)

        tmp_masked_logit = a3_logit + inf_mask

        return tmp_masked_logit

    def forward_action_model_a4(self, ctx_input, a1_action, a2_action, a3_action):

        a4_cat_input = torch.cat([ctx_input, a1_action, a2_action, a3_action], 1)

        a4_logit = self.a4_logit_layers(a4_cat_input)

        inf_mask = torch.clamp(torch.log(self.current_action_masks.get('3')), min=FLOAT_MIN)

        tmp_masked_logit = a4_logit + inf_mask

        return tmp_masked_logit


    @override(TorchModelV2)
    def forward(
            self,
            input_dict: Dict[str, TensorType],
            state: List[TensorType],
            seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):

        #if self.dict_action_masks is None:
        #self.dict_action_masks = self.create_dict_action_masks(input_dict)
        self.current_action_masks = self.store_input_action_masks(input_dict=input_dict)

        list_obs = []
        substring_action_mask = "action_mask"
        substring_action_head = "head_factor"
        # substring_trainable_disjoint_mask = "trainable_disjoint_mask"

        for key in self.obs_space_dim_dict.keys():  # also covers the case where no action mask is present
            if not re.search(substring_action_mask, key) and not re.search(substring_action_head, key):
                list_obs.append(input_dict["obs"].get(key))
        flat_inputs = torch.cat(list_obs, 1)
        obs = flat_inputs

        self._last_flat_in = obs.reshape(obs.shape[0], -1)
        self._features = self._hidden_layers(self._last_flat_in)
        # this is the context layer
        return self._features, state


    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        #if self._value_branch_separate:
        #    out = self._value_branch(
        #        self._value_branch_separate(self._last_flat_in)
        #    ).squeeze(1)
        #else:

        out = self._value_branch(self._features).squeeze(1)
        return out

    def generate_dict_gym_space_meta_coordinates(self, gym_space: gym.spaces.Dict) -> Dict:

        dict_spaces = {}
        if isinstance(gym_space, gym.spaces.Dict):
            for space_name, space in gym_space.spaces.items():
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
                    space_output_dim = int(np.sum(space.shape)) #only valid for one dimensional .Box
                elif (isinstance(space, Simplex)
                        and space is not None
                ):
                    space_output_dim = int(np.sum(space.shape))
                else:
                    raise ValueError(f'Unknown space type {space}')

                dict_spaces[space_name] = space_output_dim
            return dict_spaces
        else:
            raise ImportError(f"ToDO: account for the case the obs space is not a dict")