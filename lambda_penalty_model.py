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

class LambdaPenaltyModel(torch.nn.Module):
    def __init__(self, config_lambda_model):
        #This model outputs the summed weightes penalty terms
        super(LambdaPenaltyModel, self).__init__()

        self.availabe_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.learning_rate = config_lambda_model.get("lambda_model_lr", None)
        self.amount_constraints = config_lambda_model.get("amount_constraints", None)
        self.constraints_conditional_minkowski_encoding_type = config_lambda_model.get("constraints_conditional_minkowski_encoding_type", None)

        outputSize = 1# we sum all the weights up to get a final penalty score

        if self.constraints_conditional_minkowski_encoding_type is None: # if we do not have a conditional minkowski encoding
            self.linear = torch.nn.Linear(self.amount_constraints*2, outputSize, bias=False)
        else:
            self.linear = torch.nn.Linear(self.amount_constraints, outputSize, bias=False)

    def forward(self, x):
        out = self.linear(x)
        return out

    @staticmethod
    def update_function(param, grad, loss, learning_rate):
        zero_vector = torch.zeros_like(param)
        #print(f'Gradient: {grad}')
        #updated_value = param - learning_rate * grad
        updated_value = param + learning_rate * grad # the gradients are all positive for breaking constraints (eq+ineq)
        # and negative for being below the constraint (ineq only) -> For violations we want increased lambdas to penalize then harder

        return torch.maximum(zero_vector, updated_value) #ensure that lambda values can not be smaller than zero

    def custom_step(self):
        #implements a custom step function for this model
        # see here for detail https://discuss.pytorch.org/t/updatation-of-parameters-without-using-optimizer-step/34244/2
        # NOTE: Requires
        # model.zero_grad()
        # loss.backward()
        # -> to be called before
        with torch.no_grad():
            for p in self.parameters():
                new_val = LambdaPenaltyModel.update_function(param=p, grad=p.grad,
                                                             loss=None,
                                                             learning_rate=self.learning_rate)
                #print(new_val)
                #print("~~~~~~~~")
                p.copy_(new_val)
