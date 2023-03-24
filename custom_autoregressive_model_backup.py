import logging
import numpy as np
import gym

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, AppendBiasLayer, normc_initializer
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict

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

        layers = []
        prev_layer_size = int(np.product(obs_space.shape))
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
        """
        self.autoreg_action_branch_dict = nn.ModuleDict()
        for key, value in self.action_space_dim_dict.items():
            tmp_layers_action_branch = []
            pre_layer_size_action_branch = prev_layer_size
            # Create layers 0 to second-last.
            for size in hiddens[:-1]:
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

            self._hidden_layers = nn.Sequential(*tmp_layers_action_branch)
            #self.action_space_dim_dict.get(key)
            self.self.autoreg_action_branch_dict[key] = nn.Sequential(*tmp_layers_action_branch)
        """ or None
        ###action branches
        self.a1_logits = SlimFC(
            in_size=self.hidden_output_size,
            out_size=self.action_space_dim_dict.get("a_1"),  # self.num_outputs_action_distr, #this was 2 before
            activation_fn=None,
            initializer=normc_initializer(0.01),
        )

        self.a2_hidden = SlimFC(
            in_size=self.action_space_dim_dict.get("a_1"),
            out_size=16, #TODO fixme
            activation_fn=nn.Tanh,
            initializer=normc_initializer(0.01),
        )
        self.a2_logits = SlimFC(
            in_size=16, #TODO fixme
            out_size=self.action_space_dim_dict.get("a_2"),  # self.num_outputs_action_distr, #this was 2 before
            activation_fn=None,
            initializer=normc_initializer(0.01),
        )

    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        obs = input_dict["obs_flat"].float()
        self._last_flat_in = obs.reshape(obs.shape[0], -1)
        self._features = self._hidden_layers(self._last_flat_in)
        #this is the context layer
        return self._features, state
        #logits = self._logits(self._features) if self._logits else self._features
        #if self.free_log_std:
        #    logits = self._append_free_log_std(logits)
        #return logits, state


    def forward_action_model(self, ctx_input, a1_input):
        # WE PASS "self_" as the instance, i.e. the _ActionModel, "self" still goes on the TorchAutoregressiveActionModel
        # print("INPUT SHAPE")
        # print(ctx_input.shape)
        #print("TEST")
        #print(ctx_input.device)
        #print(a1_input.device)
        #print(next(self.parameters().device))
        #print(next(self_.parameters().device))
        #print("Finished Test")

        a1_logits = self.a1_logits(ctx_input)
        # print("INPUT A1")
        # print(a1_input.shape)
        # print(a1_input)
        a2_logits = self.a2_logits(self.a2_hidden(a1_input))
        return a1_logits, a2_logits


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