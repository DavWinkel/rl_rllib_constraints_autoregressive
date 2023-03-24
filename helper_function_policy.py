from environment_wrapper import extract_environment_class_from_config
import torch
import numpy as np
from ray.rllib.policy.sample_batch import SampleBatch
from helper_functions import generate_trainable_dataset, generate_seq_length, construct_econ_returns


def train_moment_network(policy, moment_model, train_batch,
                          torch_state_memory=None, evaluation: bool=True):

    #Check if we are in rlLib dummy run
    #if torch_state_memory[0] is not None and torch_state_memory[0].shape[2]==0:
    #    print(
    #        f"Warning: Memory has no entries with shape{torch_state_memory[0].shape}. This should only "
    #        f"occur in the very first dummy run performed by rllib")
    #    return None, None

    seq_lens = generate_seq_length(train_batch=train_batch)

    torch_econ_first_moment_return_info, torch_econ_second_moment_return_info = construct_econ_returns(moment_submodel=moment_model,
                                                                            train_batch=train_batch)

    if torch_econ_first_moment_return_info.ndim<2:
        torch_econ_first_moment_return_info = torch.unsqueeze(torch_econ_first_moment_return_info, 1) #Ensure that it is a tensor
    if torch_econ_second_moment_return_info.ndim<2:
        torch_econ_second_moment_return_info = torch.unsqueeze(torch_econ_second_moment_return_info, 1)

    input_dict = construct_moment_network_input(policy, train_batch)

    y_pred_first, y_pred_second, memory_outs = moment_model(input_dict, torch_state_memory, seq_lens)

    loss_first_moment = moment_model.loss_fn(y_pred_first, torch_econ_first_moment_return_info)
    loss_second_moment = moment_model.loss_fn(y_pred_second, torch_econ_second_moment_return_info)
    loss = loss_first_moment + loss_second_moment


    moment_model.optimizer.zero_grad()
    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()
    moment_model.optimizer.step()

    return loss_first_moment, loss_second_moment


def train_moment_network_attention_model(policy, moment_model, train_batch):

    dataset = generate_trainable_dataset(moment_submodel=moment_model, train_batch=train_batch)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

    list_pred_first = []
    list_pred_second = []
    list_obs_first = []
    list_obs_second = []
    for i, (x_batch, y_batch) in enumerate(data_loader):

        moment_model.optimizer.zero_grad()
        y_pred = moment_model(x_batch, y_batch[:, :-1])
        loss = moment_model.loss_fn(y_pred, y_batch[:, 1:])

        loss.backward()
        moment_model.optimizer.step()

        y_pred_first, y_pred_second = moment_model.split_results(y_pred.detach())
        list_pred_first.append(y_pred_first)
        list_pred_second.append(y_pred_second)
        y_obs_first, y_obs_second = moment_model.split_results(y_batch[:, 1:].detach())
        list_obs_first.append(y_obs_first)
        list_obs_second.append(y_obs_second)

    merged_pred_first = torch.cat(list_pred_first, 0)
    merged_pred_second = torch.cat(list_pred_second, 0)
    merged_obs_first = torch.cat(list_obs_first, 0)
    merged_obs_second = torch.cat(list_obs_second, 0)

    loss_first_moment = moment_model.loss_fn(merged_pred_first, merged_obs_first)
    loss_second_moment = moment_model.loss_fn(merged_pred_second, merged_obs_second)

    return loss_first_moment, loss_second_moment


def construct_moment_network_input(policy, sample_batch):
    """
    This calculated the PREVIOUSLY observed returns
    :param policy:
    :param sample_batch:
    :return:
    """
    if isinstance(sample_batch["obs"], torch.Tensor):
        is_input_tensor = True
    else:
        is_input_tensor = False

    moment_model = policy.model.moment_submodel
    print("CONSTUCT MOMENT NETWORK INPUT")
    print(moment_model.moment_model_input_type)
    input_dict_torch={}
    if moment_model.moment_model_input_type == "obs_and_action":
        if is_input_tensor:
            input_dict_torch["obs"] = torch.cat((sample_batch[SampleBatch.OBS], sample_batch[SampleBatch.ACTIONS]), 1)
        else:
            np_input = np.concatenate((sample_batch[SampleBatch.OBS], sample_batch[SampleBatch.ACTIONS]), 1)
            input_dict_torch["obs"] = torch.from_numpy(np_input).float()
    elif moment_model.moment_model_input_type == "obs":
        if is_input_tensor:
            input_dict_torch["obs"] = sample_batch[SampleBatch.OBS]
        else:
            np_input = sample_batch[SampleBatch.OBS]
            input_dict_torch["obs"] = torch.from_numpy(np_input).float()
    elif moment_model.moment_model_input_type == "only_prev_returns":
        if is_input_tensor:
            np_input = sample_batch[SampleBatch.OBS].cpu().detach().numpy()
        else:
            np_input = sample_batch[SampleBatch.OBS]
        environment_class = extract_environment_class_from_config(policy.config)
        dict_decomposed = environment_class.decompose_observation_wrapper_dict(np_input,
                                                                               config=policy.config
                                                                               #include_risk_penalty_in_state=
                                                                               #policy.config.get("env_config").get("include_risk_penalty_in_state"),
                                                                               )
        prev_returns = dict_decomposed["prev_observed_returns"]
        input_dict_torch["obs"] = torch.from_numpy(prev_returns).float()
    else:
        raise ValueError("Unknown moment input type")
    return input_dict_torch
