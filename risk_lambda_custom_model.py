import logging
import numpy as np
import gym

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, AppendBiasLayer, normc_initializer
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.catalog import ModelCatalog
from lambda_penalty_model import LambdaPenaltyModel
from moment_model import MomentModel
from moment_model_transformer import MomentTransformerModel


torch, nn = try_import_torch()

logger = logging.getLogger(__name__)

from helper_functions import calculate_action_dim

class RiskLambdaCustomModel(TorchModelV2, nn.Module):
    """Generic fully connected network."""

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

        self.number_assets = int(np.product(action_space.shape))
        print("We are in the risk lambda custom model")

        hiddens = list(model_config.get("fcnet_hiddens", [])) + list(
            model_config.get("post_fcnet_hiddens", [])
        )
        activation = model_config.get("fcnet_activation")
        if not model_config.get("fcnet_hiddens", []):
            activation = model_config.get("post_fcnet_activation")
        no_final_linear = model_config.get("no_final_linear")
        self.vf_share_layers = model_config.get("vf_share_layers")
        self.free_log_std = model_config.get("free_log_std")
        # Generate free-floating bias variables for the second half of
        # the outputs.
        if self.free_log_std:
            assert num_outputs % 2 == 0, (
                "num_outputs must be divisible by two",
                num_outputs,
            )
            num_outputs = num_outputs // 2

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
        if self.free_log_std and self._logits:
            self._append_free_log_std = AppendBiasLayer(num_outputs)

        self._hidden_layers = nn.Sequential(*layers)

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

        # Adding a lambda penalty model
        # we need to specify the amount of constraints
        # we could calculate in the config_helper and add it as a parameter for the model config
        config_lambda_model = model_config.get("custom_model_config").get("config_lambda_model", None)

        self.lambda_penalty_model = LambdaPenaltyModel(config_lambda_model=config_lambda_model)


        #Optional Risk part
        # Introduce risk model
        if "custom_model_config" in model_config and "config_moment_model" in model_config.get(
                "custom_model_config"):

            number_input_assets = self.number_assets
            moment_model_config = model_config.get("custom_model_config").get("config_moment_model")
            self.moment_model_attention_dim = model_config.get("custom_model_config").get(
                "config_moment_model").get(
                "attention_dim")

            self.use_moment_attention = model_config.get("custom_model_config").get("config_moment_model").get(
                "use_moment_attention")

            if self.use_moment_attention:
                self.moment_submodel = MomentTransformerModel(config_attention_model=moment_model_config,
                                                              in_space=obs_space,
                                                              number_input_assets=number_input_assets)  # action_space=action_space)
            else:
                self.moment_submodel = MomentModel(config_moment_model=moment_model_config,
                                                   in_space=obs_space, action_space=action_space)
        else:
            print("No moment model currently used")

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
        logits = self._logits(self._features) if self._logits else self._features
        if self.free_log_std:
            logits = self._append_free_log_std(logits)
        return logits, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        if self._value_branch_separate:
            return self._value_branch(
                self._value_branch_separate(self._last_flat_in)
            ).squeeze(1)
        else:
            return self._value_branch(self._features).squeeze(1)

    def test(self):
        print("THIS FUNCTION IS REACHABLE")


def make_risk_model(
        policy: TorchPolicyV2, obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        config: Dict) -> ModelV2:

    print("Making RISKLAMBDA MODEL")
    print(f'Model obs space {obs_space}')
    print(f'Model action space {action_space}')
    #print("--------")
    #if isinstance(action_space, gym.spaces.Box):
    #    #-> for box environment ppo we use normal distribution, i.e. 2 parameter per value
    #    num_outputs = action_space.shape[0]*2##action_space.n
    #else:
    #    num_outputs = action_space.dim

    num_outputs = calculate_action_dim(action_space)

    model = ModelCatalog.get_model_v2(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=num_outputs,#action_space.n,
        model_config=config["model"],
        framework=config["framework"],
        # Providing the `model_interface` arg will make the factory
        # wrap the chosen default model with our new model API class
        # (DummyCustomModel). This way, both `forward` and `get_q_values`
        # are available in the returned class.
        model_interface=RiskLambdaCustomModel,
        name="risk_lambda_model",
    )

    return model
