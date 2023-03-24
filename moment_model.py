from ray.rllib.models.torch.attention_net import GTrXLNet
import torch
from typing import Dict, Optional, Union
from ray.rllib.utils.framework import try_import_torch
import gym
import numpy as np

from ray.rllib.utils.typing import ModelConfigDict, TensorType, List

torch, nn = try_import_torch()
from ray.rllib.models.torch.misc import SlimFC, AppendBiasLayer, \
    normc_initializer

from ray.rllib.models.torch.misc import SlimFC

class MomentModel(torch.nn.Module):

    def __init__(self, config_moment_model, in_space, action_space):

        super(MomentModel, self).__init__()

        self.availabe_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.config = config_moment_model
        self.action_space = action_space

        self.moment_model_input_type = config_moment_model["moment_model_input_type"]
        self.moment_model_output_aggregated_portfolio = config_moment_model["moment_model_output_aggregated_portfolio"]

        self.moment_model_lr = config_moment_model["moment_model_lr"]

        self.moment_model_hiddens = config_moment_model["moment_model_hiddens"]
        activation = config_moment_model["moment_model_hidden_activation"]

        layers = []

        amount_assets = action_space.shape[0]
        if config_moment_model["moment_model_output_aggregated_portfolio"]:
            self.num_outputs_first_moment = 1
            self.num_outputs_second_moment = 1
        else:
            self.num_outputs_first_moment = amount_assets  # output_size
            # This is true for the covariance matrix
            self.num_outputs_second_moment = int((amount_assets + 1) * (amount_assets / 2))  # N + (N+1)*N/2

        cfg = config_moment_model

        if self.moment_model_input_type=="obs_and_action":
            in_space_dim = in_space.shape[0] + action_space.shape[0]
            in_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(
                in_space_dim,))
        elif self.moment_model_input_type=="obs":
            pass
        elif self.moment_model_input_type=="only_prev_returns":
            in_space_dim = amount_assets
            in_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(
                in_space_dim,))

        prev_layer_size = in_space.shape[0]

        # Create layers 0 to second-last.
        for size in self.moment_model_hiddens[:-1]:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=size,
                    initializer=normc_initializer(1.0),
                    activation_fn=activation))
            prev_layer_size = size
        # create last layer
        if len(self.moment_model_hiddens) > 0:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=self.moment_model_hiddens[-1],
                    initializer=normc_initializer(1.0),
                    activation_fn=activation))
            prev_layer_size = self.moment_model_hiddens[-1]

        self._hidden_layers = torch.nn.Sequential(*layers)

        amount_possible_states = 1
        self._logit_softmax_input = torch.nn.Sequential(SlimFC(
            in_size=prev_layer_size,
            out_size=amount_possible_states,
            activation_fn=None,
            initializer=torch.nn.init.xavier_uniform_))

        self.softmax_layer = torch.nn.Softmax(dim=1)

        self._logits_branch_first_moment = SlimFC(
            in_size=amount_possible_states,
            out_size=self.num_outputs_first_moment,
            activation_fn=None,
            use_bias=False,
            initializer=torch.nn.init.xavier_uniform_)

        self._logits_branch_second_moment = SlimFC(
            in_size=amount_possible_states,
            out_size=self.num_outputs_second_moment,
            activation_fn=None,
            use_bias=False,
            initializer=torch.nn.init.xavier_uniform_)


        #self.optimizer = torch.optim.Adam(self.parameters(), lr=self.moment_model_lr)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.moment_model_lr)
        # loss
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, TensorType, List[TensorType]):

        input_dict["obs"] = input_dict["obs"].to(self.availabe_device)

        self._features = self._hidden_layers(input_dict["obs"])

        self._logit_softmax = self._logit_softmax_input(self._features)

        self._softmax_output = self.softmax_layer(self._logit_softmax)

        model_out_first_moment = self._logits_branch_first_moment(self._softmax_output)
        model_out_second_moment = self._logits_branch_second_moment(self._softmax_output)

        memory_outs = None

        return model_out_first_moment, model_out_second_moment, memory_outs
