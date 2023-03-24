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

torch, nn = try_import_torch()

logger = logging.getLogger(__name__)


class TorchCustomAutoregressiveModel(TorchModelV2, nn.Module):
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

        self.action_space_dim_dict = self.generate_action_space_dim_dict()

        activation = model_config.get("fcnet_activation")
        #if not model_config.get("fcnet_hiddens", []):
        #    activation = model_config.get("post_fcnet_activation")
        #no_final_linear = model_config.get("no_final_linear")
        self.vf_share_layers = model_config.get("vf_share_layers")
        #self.free_log_std = model_config.get("free_log_std")
        # Generate free-floating bias variables for the second half of
        # the outputs.
        #if self.free_log_std:
        #    assert num_outputs % 2 == 0, (
        #        "num_outputs must be divisible by two",
        #        num_outputs,
        #    )
        #    num_outputs = num_outputs // 2

        #print("OBS SPACE")
        #print(obs_space)

        substring_action_mask = "action_mask"
        substring_action_head = "head_factor"

        self.obs_size = 0
        for key, value in self.obs_space_dim_dict.items():
            if not re.search(substring_action_mask, key) and not re.search(substring_action_head, key):
                self.obs_size += value

        layers = []
        prev_layer_size = self.obs_size #int(np.product(obs_space.shape))
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

        #generate last layer for context layer
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

        """
        # The last layer is adjusted to be of size num_outputs, but it's a
        # layer with activation.
        if no_final_linear and num_outputs:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=num_outputs,
                    initializer=normc_initializer(1.0),
                    activation_fn=activation,
                )
            )
            prev_layer_size = num_outputs
        # Finish the layers with the provided sizes (`hiddens`), plus -
        # iff num_outputs > 0 - a last linear layer of size num_outputs.
        else:
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
            if num_outputs:
                self._logits = SlimFC(
                    in_size=prev_layer_size,
                    out_size=num_outputs,
                    initializer=normc_initializer(0.01),
                    activation_fn=None,
                )
            else:
                self.num_outputs = ([int(np.product(obs_space.shape))] + hiddens[-1:])[
                    -1
                ]

        # Layer to add the log std vars to the state-dependent means.
        #if self.free_log_std and self._logits:
        #    self._append_free_log_std = AppendBiasLayer(num_outputs)
        """ or None

        self._hidden_layers = nn.Sequential(*layers)

        """
        self._value_branch_separate = None
        if not self.vf_share_layers:
            # Build a parallel set of hidden layers for the value net.
            prev_vf_layer_size = int(np.product(obs_space.shape))
            vf_layers = []
            for size in hiddens:
                vf_layers.append(
                    SlimFC(
                        in_size=prev_vf_layer_size,
                        out_size=size,
                        activation_fn=activation,
                        initializer=normc_initializer(1.0),
                    )
                )
                prev_vf_layer_size = size
            self._value_branch_separate = nn.Sequential(*vf_layers)
        """ or None
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


        #dict for model
        if "custom_model_config" in model_config:
            hiddens_action_branch = list(model_config.get("custom_model_config").get("fcnet_hiddens_autoreg_branches", []))

        self.autoreg_action_branch_dict = nn.ModuleDict()
        self.autoreg_input_size = 0
        for key, value in self.action_space_dim_dict.items():
            tmp_layers_action_branch = []
            pre_layer_size_action_branch = prev_layer_size + self.autoreg_input_size
            # Create layers 0 to second-last.
            for size in hiddens_action_branch[:-1]:
                tmp_layers_action_branch.append(
                    SlimFC(
                        in_size=pre_layer_size_action_branch,
                        out_size=size,
                        initializer=normc_initializer(1.0),
                        activation_fn=activation,
                    )
                )
                pre_layer_size_action_branch = size
            
            #add final logit layer
            tmp_layers_action_branch.append(
                SlimFC(
                in_size=pre_layer_size_action_branch,  # TODO fixme
                out_size=self.action_space_dim_dict.get(key),  # self.num_outputs_action_distr, #this was 2 before
                activation_fn=None,
                initializer=normc_initializer(0.01),
                )
            )
            #self.action_space_dim_dict.get(key)
            self.autoreg_action_branch_dict[key] = nn.Sequential(*tmp_layers_action_branch)
            self.autoreg_input_size += value

        self.dict_action_masks = None

    def create_dict_action_masks(self, input_dict):
        substring_action_mask = "action_mask"
        dict_action_mask = {}
        for key in self.obs_space_dim_dict.keys():
            if re.search(substring_action_mask, key):
                mask_idx = key.split("_")[0]
                tmp_action_mask = input_dict["obs"].get(key)
                tmp_action_mask = tmp_action_mask[0]
                #print(tmp_action_mask.shape)
                #print("---")
                dict_action_mask[mask_idx] = tmp_action_mask #input_dict["obs"].get(key)
        return dict_action_mask

    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):

        if self.dict_action_masks is None:
            self.dict_action_masks = self.create_dict_action_masks(input_dict)

        list_obs = []
        substring_action_mask = "action_mask"
        substring_action_head = "head_factor"
        #substring_trainable_disjoint_mask = "trainable_disjoint_mask"

        for key in self.obs_space_dim_dict.keys():  # also covers the case where no action mask is present
            if not re.search(substring_action_mask, key) and not re.search(substring_action_head, key):
                list_obs.append(input_dict["obs"].get(key))
        flat_inputs = torch.cat(list_obs, 1)
        obs = flat_inputs
        #obs = input_dict["obs_flat"].float()

        self._last_flat_in = obs.reshape(obs.shape[0], -1)
        self._features = self._hidden_layers(self._last_flat_in)
        #this is the context layer
        return self._features, state
        #logits = self._logits(self._features) if self._logits else self._features
        #if self.free_log_std:
        #    logits = self._append_free_log_std(logits)
        #return logits, state


    def forward_action_model(self, ctx_input, list_action_inputs):

        list_outputs = []
        BATCH_SIZE = ctx_input.shape[0]
        for ctx, (key, value) in enumerate(self.action_space_dim_dict.items()):

            if ctx==0:
                forward_input = ctx_input
            else:
                if ctx==1:
                    forward_input = torch.cat([ctx_input, list_action_inputs[(ctx-1)]], 1)
                else:
                    #TODO UNTESTED
                    forward_input = torch.cat([ctx_input, *list_action_inputs[:(ctx-1)]], 1)
            tmp_logit = self.autoreg_action_branch_dict[key](forward_input)
            action_mask_merged = self.dict_action_masks.get(str(ctx)).repeat(BATCH_SIZE, 1)
            inf_mask = torch.clamp(torch.log(action_mask_merged), min=FLOAT_MIN)
            tmp_masked_logit = tmp_logit + inf_mask
            list_outputs.append(tmp_masked_logit)

        return list_outputs
            #self.autoreg_action_branch_dict[
            #a2_context = tf.keras.layers.Concatenate(axis=1)(
            #     [ctx_input, a1_input])
        #a1_logits = self.a1_logits(ctx_input)
        # print("INPUT A1")
        # print(a1_input.shape)
        # print(a1_input)
        #a2_logits = self.a2_logits(self.a2_hidden(a1_input))
        #return a1_logits, a2_logits


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

    def generate_action_space_dim_dict(self):
        action_space_dim_dict = {}
        if isinstance(self.action_space, gym.spaces.Dict):

            for key, value in self.action_space.spaces.items():
                action_space_dim_dict[key] = value.shape[0]

        return action_space_dim_dict


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
