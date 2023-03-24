from financial_markets_gym.envs.financial_markets_env import FinancialMarketsEnv
from ray.tune.registry import register_env
from environment_wrapper import BasicWrapperFinancialEnvironmentEvaluationable, BasicWrapperFinancialEnvironmentShortSelling
from ray.rllib.models.catalog import ModelCatalog
import torch
from datetime import datetime
from typing import Dict, Tuple, Union, Any
from ray.rllib.utils.typing import TensorType, List, Union, Tuple, ModelConfigDict
from ray.air import CheckpointConfig

import yaml

def create_env_config(run_config_raw, yaml_file_name):

    run_name = yaml_file_name.split(".yaml")[0]

    run_config_readable = convert_yaml_into_readable_format(run_config_input=run_config_raw)
    run_config_readable = convert_risk_penalty_format(run_config_readable)
    run_config_readable = register_trainer_from_config(run_config_readable)
    run_config_readable = register_custom_env(run_config_readable)
    run_config_readable = register_custom_action_distribution(run_config_readable)
    run_config_readable = register_custom_model(run_config_readable)
    run_config_readable = update_GPU_resources(run_config_readable)
    run_config_readable = check_for_custom_eval_function(run_config_readable)
    run_config_readable = check_for_custom_callbacks(run_config_readable)
    run_config_readable = check_for_non_config_callbacks(run_config_readable, group_name=run_name)

    return run_config_readable.get("config").get("env_config")

def convert_runable_into_2_0_format(run_config_input: Dict)-> Tuple[Union[str, Any], Dict, Dict]:

    trainable = run_config_input.pop("run_or_experiment")
    dict_param_space = run_config_input.pop("config")
    remaining_run_config_input = run_config_input

    if "checkpoint_config" in run_config_input:
        run_config_input["checkpoint_config"] = CheckpointConfig(**run_config_input.pop("checkpoint_config"))

    return trainable, dict_param_space, remaining_run_config_input

def convert_yaml_into_readable_format(run_config_input):
    if "config" not in run_config_input:
        run_config_input["config"] = {}
    if "env" in run_config_input:
        run_config_input["config"]["env"] = run_config_input.pop("env")
    if "run" in run_config_input:
        run_config_input["run_or_experiment"] = run_config_input.pop("run")
    return run_config_input

def convert_risk_penalty_format(run_config_input):

    if ("risk_penalty_factor_lower_bound" in run_config_input["config"]) & ("risk_penalty_factor_upper_bound" in run_config_input["config"]):
        run_config_input["config"]["env_config"]["risk_penalty_factor_lower_bound"] = run_config_input["config"].pop(
            "risk_penalty_factor_lower_bound")
        run_config_input["config"]["env_config"]["risk_penalty_factor_upper_bound"] = run_config_input["config"].pop(
            "risk_penalty_factor_upper_bound")

    if ("risk_penalty_factor" in run_config_input["config"]):
        run_config_input["config"]["env_config"]["risk_penalty_factor"] = run_config_input["config"].pop("risk_penalty_factor") # here we just copy the value
    if ("include_risk_penalty_in_state" in run_config_input["config"]):
        run_config_input["config"]["env_config"]["include_risk_penalty_in_state"] = run_config_input["config"].pop(
            "include_risk_penalty_in_state")  # here we just copy the value



    return run_config_input


def register_custom_env(run_config_input):
    #need to check first if there is an env_config_id used
    run_config_input = check_for_env_config_id(run_config_input)

    # only applies to RISKPPOs so far -> wrapping in a metapolicy that allows for single time step evalation
    select_env = run_config_input.get("config").get("env")
    print("ENV CONFIG!")
    print(run_config_input.get(
                "config").get(
                "env_config"))
    if select_env == "financial-markets-env-v0":
        select_env_wrapped = f'wrapped-{select_env}'
        run_config_input["config"]["env"] = select_env_wrapped
        register_env(select_env_wrapped, lambda config: BasicWrapperFinancialEnvironmentEvaluationable(
            FinancialMarketsEnv(**config),
            config=config))
        print(f"Successfully installed {select_env_wrapped}")
    elif select_env == "financial-markets-env-short-selling-v0":
        select_env_wrapped = f'wrapped-{select_env}'
        run_config_input["config"]["env"] = select_env_wrapped
        register_env(select_env_wrapped, lambda config: BasicWrapperFinancialEnvironmentShortSelling(
            FinancialMarketsEnv(**config),
            config=config))
        print(f"Successfully installed {select_env_wrapped}")
    elif select_env == "stateless_cartpole":
        from ray.rllib.examples.env.stateless_cartpole import StatelessCartPole
        register_env("stateless_cartpole", lambda config: StatelessCartPole(config))
    elif select_env == "CorrelatedActionsEnv":
        from ray.rllib.examples.env.correlated_actions_env import CorrelatedActionsEnv
        register_env("CorrelatedActionsEnv", lambda config: CorrelatedActionsEnv(config))
    elif select_env == "CustomCorrelatedActionsEnv":
        print("PICKED CUSTOM CORRELATED ACTIONS")
        from testing_env import CustomCorrelatedActionsEnv
        register_env("CustomCorrelatedActionsEnv", lambda config: CustomCorrelatedActionsEnv(config))
    elif select_env == "CustomCorrelatedActionsDirichletEnv":
        from testing_env import CustomCorrelatedActionsDirichletEnv
        register_env("CustomCorrelatedActionsDirichletEnv", lambda  config: CustomCorrelatedActionsDirichletEnv(config))
    return run_config_input

def register_custom_action_distribution(run_config_input):

    if "custom_action_dist" in run_config_input.get("config").get("model", []):
        custom_dist_name=run_config_input.get("config").get("model", []).get("custom_action_dist", [])

        if custom_dist_name == "dirichlet_dist_custom":
            from dirichlet_custom import TorchDirichlet_Custom
            ModelCatalog.register_custom_action_dist("dirichlet_dist_custom", TorchDirichlet_Custom)
            print("REGISTERED DIRICHLET CUSTOM")
        if custom_dist_name == "multi_action_distribution_custom":
            from distribution_custom import TorchMultiActionDistribution_Custom
            ModelCatalog.register_custom_action_dist("multi_action_distribution_custom", TorchMultiActionDistribution_Custom)
        if custom_dist_name == "binary_autoreg_dist":
            from distribution_autoregressive_custom import TorchBinaryAutoregressiveDistribution
            ModelCatalog.register_custom_action_dist("binary_autoreg_dist",
                                                     TorchBinaryAutoregressiveDistribution)
        if custom_dist_name == "My_betadist":
            from distribution_autoregressive_custom import My_betadist
            ModelCatalog.register_custom_action_dist("My_betadist",
                                                     My_betadist)
        if custom_dist_name == "TorchAutoregressiveDirichletDistribution":
            from distribution_autoregressive_custom import TorchAutoregressiveDirichletDistribution
            ModelCatalog.register_custom_action_dist("TorchAutoregressiveDirichletDistribution",
                                                     TorchAutoregressiveDirichletDistribution)
        if custom_dist_name == "TorchAutoregressiveDirichletDistributionV2":
            from distribution_autoregressive_custom import TorchAutoregressiveDirichletDistributionV2
            ModelCatalog.register_custom_action_dist("TorchAutoregressiveDirichletDistributionV2",
                                                     TorchAutoregressiveDirichletDistributionV2)

        if custom_dist_name == "TorchAutoregressiveDirichletDistributionTypeOne":
            from distribution_autoregressive_custom_type_based import TorchAutoregressiveDirichletDistributionTypeOne
            ModelCatalog.register_custom_action_dist("TorchAutoregressiveDirichletDistributionTypeOne",
                                                     TorchAutoregressiveDirichletDistributionTypeOne)
        if custom_dist_name == "TorchAutoregressiveDirichletDistributionTypeOneTestingOne":
            from distribution_autoregressive_custom_type_based import TorchAutoregressiveDirichletDistributionTypeOneTestingOne
            ModelCatalog.register_custom_action_dist("TorchAutoregressiveDirichletDistributionTypeOneTestingOne",
                                                     TorchAutoregressiveDirichletDistributionTypeOneTestingOne)
        if custom_dist_name == "TorchAutoregressiveDirichletDistributionTypeOneTestingTwo":
            from distribution_autoregressive_custom_type_based import TorchAutoregressiveDirichletDistributionTypeOneTestingTwo
            ModelCatalog.register_custom_action_dist("TorchAutoregressiveDirichletDistributionTypeOneTestingTwo",
                                                     TorchAutoregressiveDirichletDistributionTypeOneTestingTwo)
        if custom_dist_name == "TorchAutoregressiveDirichletDistributionTypeOneTestingThree":
            from distribution_autoregressive_custom_type_based import TorchAutoregressiveDirichletDistributionTypeOneTestingThree
            ModelCatalog.register_custom_action_dist("TorchAutoregressiveDirichletDistributionTypeOneTestingThree",
                                                     TorchAutoregressiveDirichletDistributionTypeOneTestingThree)

        if custom_dist_name == "TorchAutoregressiveDirichletDistributionTypeOneTestingFour":
            from distribution_autoregressive_custom_type_based import TorchAutoregressiveDirichletDistributionTypeOneTestingFour
            ModelCatalog.register_custom_action_dist("TorchAutoregressiveDirichletDistributionTypeOneTestingFour",
                                                     TorchAutoregressiveDirichletDistributionTypeOneTestingFour)
        if custom_dist_name == "TorchAutoregressiveDirichletDistributionTypeOneTestingFive":
            from distribution_autoregressive_custom_type_based import TorchAutoregressiveDirichletDistributionTypeOneTestingFive
            ModelCatalog.register_custom_action_dist("TorchAutoregressiveDirichletDistributionTypeOneTestingFive",
                                                     TorchAutoregressiveDirichletDistributionTypeOneTestingFive)
        if custom_dist_name == "TorchAutoregressiveDirichletDistributionTypeOneTestingSix":
            from distribution_autoregressive_custom_type_based import TorchAutoregressiveDirichletDistributionTypeOneTestingSix
            ModelCatalog.register_custom_action_dist("TorchAutoregressiveDirichletDistributionTypeOneTestingSix",
                                                     TorchAutoregressiveDirichletDistributionTypeOneTestingSix)
        if custom_dist_name == "TorchAutoregressiveDirichletDistributionTypeOneTestingSeven":
            from distribution_autoregressive_custom_type_based import TorchAutoregressiveDirichletDistributionTypeOneTestingSeven
            ModelCatalog.register_custom_action_dist("TorchAutoregressiveDirichletDistributionTypeOneTestingSeven",
                                                     TorchAutoregressiveDirichletDistributionTypeOneTestingSeven)
        if custom_dist_name == "TorchAutoregressiveDirichletDistributionTypeOneTestingEight":
            from distribution_autoregressive_custom_type_based import TorchAutoregressiveDirichletDistributionTypeOneTestingEight
            ModelCatalog.register_custom_action_dist("TorchAutoregressiveDirichletDistributionTypeOneTestingEight",
                                                     TorchAutoregressiveDirichletDistributionTypeOneTestingEight)
        if custom_dist_name == "TorchAutoregressiveDirichletDistributionTypeOneTestingNine":
            from distribution_autoregressive_custom_type_based import TorchAutoregressiveDirichletDistributionTypeOneTestingNine
            ModelCatalog.register_custom_action_dist("TorchAutoregressiveDirichletDistributionTypeOneTestingNine",
                                                     TorchAutoregressiveDirichletDistributionTypeOneTestingNine)
        if custom_dist_name == "TorchAutoregressiveDirichletDistributionS2":
            from distribution_autoregressive_custom_types import TorchAutoregressiveDirichletDistributionS2
            ModelCatalog.register_custom_action_dist("TorchAutoregressiveDirichletDistributionS2",
                                                     TorchAutoregressiveDirichletDistributionS2)
        if custom_dist_name == "TorchAutoregressiveDirichletDistributionS3":
            from distribution_autoregressive_custom_types import TorchAutoregressiveDirichletDistributionS3
            ModelCatalog.register_custom_action_dist("TorchAutoregressiveDirichletDistributionS3",
                                                     TorchAutoregressiveDirichletDistributionS3)
        if custom_dist_name == "TorchAutoregressiveDirichletDistributionS4":
            from distribution_autoregressive_custom_types import TorchAutoregressiveDirichletDistributionS4
            ModelCatalog.register_custom_action_dist("TorchAutoregressiveDirichletDistributionS4",
                                                     TorchAutoregressiveDirichletDistributionS4)
        if custom_dist_name == "TorchAutoregressiveDirichletDistributionS4_U1":
            from distribution_autoregressive_custom_types import TorchAutoregressiveDirichletDistributionS4_U1
            ModelCatalog.register_custom_action_dist("TorchAutoregressiveDirichletDistributionS4_U1",
                                                     TorchAutoregressiveDirichletDistributionS4_U1)
        if custom_dist_name == "TorchBaselinePolytopeDistribution":
            from distribution_baseline_custom import TorchBaselinePolytopeDistribution
            ModelCatalog.register_custom_action_dist("TorchBaselinePolytopeDistribution",
                                                     TorchBaselinePolytopeDistribution)

    return run_config_input

def register_custom_model(run_config_input):

    if "custom_model" in run_config_input.get("config").get("model", []):
        custom_model_name=run_config_input.get("config").get("model", []).get("custom_model")

        if custom_model_name == "risk_model_ppo":
            from risk_ppo_custom_model import CustomPPOTorchModel
            ModelCatalog.register_custom_model("risk_model_ppo", CustomPPOTorchModel)
        elif custom_model_name == "risk_model_ddpg":
            from risk_ddpg_custom_model import CustomDDPGTorchModel
            ModelCatalog.register_custom_model("risk_model_ddpg", CustomDDPGTorchModel)
        elif custom_model_name == "MaskedTorchFCRNNModel":
            from custom_model_masking import MaskedTorchFCRNNModel
            ModelCatalog.register_custom_model("MaskedTorchFCRNNModel", MaskedTorchFCRNNModel)
        elif custom_model_name == "RiskLambdaCustomModel":
            from risk_lambda_custom_model import RiskLambdaCustomModel
            ModelCatalog.register_custom_model("RiskLambdaCustomModel", RiskLambdaCustomModel)
        elif custom_model_name == "CostValueFunctionCustomModel":
            from risk_cost_vf_custom_model import CostValueFunctionCustomModel
            ModelCatalog.register_custom_model("CostValueFunctionCustomModel", CostValueFunctionCustomModel)
        elif custom_model_name =="autoregressive_model":
            from model_autoregressive_custom import TorchAutoregressiveActionModel
            ModelCatalog.register_custom_model("autoregressive_model", TorchAutoregressiveActionModel)
        elif custom_model_name == "autoregressive_model_v2":
            from model_autoregressive_custom import TorchAutoregressiveActionModelV2
            ModelCatalog.register_custom_model("autoregressive_model_v2", TorchAutoregressiveActionModelV2)
        elif custom_model_name == "TorchCustomAutoregressiveModel":
            from custom_autoregressive_model import TorchCustomAutoregressiveModel
            ModelCatalog.register_custom_model("TorchCustomAutoregressiveModel", TorchCustomAutoregressiveModel)
        elif custom_model_name == "TorchCustomAutoregressiveModelTypeOne":
            from custom_autoregressive_model_type_based import TorchCustomAutoregressiveModelTypeOne
            ModelCatalog.register_custom_model("TorchCustomAutoregressiveModelTypeOne", TorchCustomAutoregressiveModelTypeOne)
        elif custom_model_name == "TorchCustomAutoregressiveModelTypeTwo":
            from custom_autoregressive_model_type_based import TorchCustomAutoregressiveModelTypeTwo
            ModelCatalog.register_custom_model("TorchCustomAutoregressiveModelTypeTwo",
                                               TorchCustomAutoregressiveModelTypeTwo)
        elif custom_model_name == "TorchCustomAutoregressiveModelTypeThree":
            from custom_autoregressive_model_type_based import TorchCustomAutoregressiveModelTypeThree
            ModelCatalog.register_custom_model("TorchCustomAutoregressiveModelTypeThree",
                                               TorchCustomAutoregressiveModelTypeThree)
        elif custom_model_name == "TorchCustomAutoregressiveModelTypeFour":
            from custom_autoregressive_model_type_based import TorchCustomAutoregressiveModelTypeFour
            ModelCatalog.register_custom_model("TorchCustomAutoregressiveModelTypeFour",
                                               TorchCustomAutoregressiveModelTypeFour)
        elif custom_model_name == "TorchCustomAutoregressiveModelTypeFive":
            from custom_autoregressive_model_type_based import TorchCustomAutoregressiveModelTypeFive
            ModelCatalog.register_custom_model("TorchCustomAutoregressiveModelTypeFive",
                                               TorchCustomAutoregressiveModelTypeFive)
        elif custom_model_name == "TorchCustomAutoregressiveModelS2":
            from custom_autoregressive_model_S2 import TorchCustomAutoregressiveModelS2
            ModelCatalog.register_custom_model("TorchCustomAutoregressiveModelS2",
                                               TorchCustomAutoregressiveModelS2)
        elif custom_model_name == "TorchCustomAutoregressiveModelS3":
            from custom_autoregressive_model_S3 import TorchCustomAutoregressiveModelS3
            ModelCatalog.register_custom_model("TorchCustomAutoregressiveModelS3",
                                               TorchCustomAutoregressiveModelS3)
        elif custom_model_name == "TorchCustomAutoregressiveModelS4":
            from custom_autoregressive_model_S4 import TorchCustomAutoregressiveModelS4
            ModelCatalog.register_custom_model("TorchCustomAutoregressiveModelS4",
                                               TorchCustomAutoregressiveModelS4)
        elif custom_model_name == "TorchCustomAutoregressiveModelS4_U1":
            from custom_autoregressive_model_S4_U1 import TorchCustomAutoregressiveModelS4_U1
            ModelCatalog.register_custom_model("TorchCustomAutoregressiveModelS4_U1",
                                               TorchCustomAutoregressiveModelS4_U1)
        elif custom_model_name == "CustomBaselineModel":
            from custom_baseline_model import CustomBaselineModel
            ModelCatalog.register_custom_model("CustomBaselineModel",
                                               CustomBaselineModel)
            # special requirement for this model type to include the polytope data into the custom model, since it is important information
            dict_constraint_data = {
                'head_factor_list': run_config_input.get("config").get("env_config").get("head_factor_list"),
                'action_mask_dict': run_config_input.get("config").get("env_config").get("action_mask_dict")
            }
            if "custom_model_config" in run_config_input.get("config").get("model"):
                run_config_input.get("config").get("model").get("custom_model_config")[
                    "constraint_data"] = dict_constraint_data
                run_config_input.get("config").get("model").get("custom_model_config")[
                    "constraints_conditional_minkowski_encoding_type"] = \
                    run_config_input.get("config").get("env_config").get("constraints_conditional_minkowski_encoding_type")
            else:
                raise ValueError("No custom_model_config present")

        print(f"REGISTERED MODEL {custom_model_name}")

    return run_config_input

def register_algorithm_from_config(run_config_readable):

    algorithm_name = run_config_readable.get("run_or_experiment")

    if algorithm_name == "RiskShortPPO":
        from risk_short_ppo_algorithm import RiskShortPPO
        run_config_readable["run_or_experiment"] = RiskShortPPO
        return run_config_readable
    elif algorithm_name == "RiskLagrangePPO":
        from risk_lagrange_ppo_algorithm import RiskLagrangePPO
        run_config_readable["run_or_experiment"] = RiskLagrangePPO
        return run_config_readable
    elif algorithm_name == "RiskIPOPPO":
        from risk_ipo_ppo_algorithm import RiskIPOPPO
        run_config_readable["run_or_experiment"] = RiskIPOPPO
        return run_config_readable
    elif algorithm_name == "P3OPPO":
        from risk_P3O_ppo_algorithm import P3OPPO
        run_config_readable["run_or_experiment"] = P3OPPO
        return run_config_readable
    elif algorithm_name == "RiskIPOMergedPPO":
        from risk_ipo_ppo_algorithm_merged import RiskIPOMergedPPO
        run_config_readable["run_or_experiment"] = RiskIPOMergedPPO
        return run_config_readable
    elif algorithm_name == "RiskAutoregressivePPO":
        from risk_autoregressive_ppo_algorithm import RiskAutoregressivePPO
        run_config_readable["run_or_experiment"] = RiskAutoregressivePPO
        return run_config_readable
    elif algorithm_name == "Baseline":
        from risk_baseline_algorithm import Baseline
        run_config_readable["run_or_experiment"] = Baseline
        return run_config_readable
    else:
        print("WARNING: No custom algo found")
        return run_config_readable

def register_trainer_from_config(run_config_readable):
    """
    #DEPRECITATED Trainers turned into algorithms in rllib 2.0
    Necessary for the custom trainers due to missing reference in an init file
    :param trainer_name:
    :return:
    """
    trainer_name = run_config_readable.get("run_or_experiment")

    if trainer_name == "RiskPPO":
        from risk_ppo_trainer import RiskPPOTrainer
        run_config_readable["run_or_experiment"] = RiskPPOTrainer
        return run_config_readable
    elif trainer_name == "RiskDDPG":
        from risk_ddpg_trainer import RiskDDPGTrainer
        run_config_readable["run_or_experiment"] = RiskDDPGTrainer
        return run_config_readable
    elif trainer_name == "RiskTD3":
        from risk_td3_trainer import RiskTD3Trainer
        run_config_readable["run_or_experiment"] = RiskTD3Trainer
        return run_config_readable
    elif trainer_name == "RiskShortPPO":
        from risk_short_ppo_trainer import RiskShortPPOTrainer
        run_config_readable["run_or_experiment"] = RiskShortPPOTrainer
        return run_config_readable
    elif trainer_name == "NonRiskShortPPO":
        from non_risk_short_ppo_trainer import  NonRiskShortPPOTrainer
        run_config_readable["run_or_experiment"] = NonRiskShortPPOTrainer
        return run_config_readable
    else:
        return run_config_readable

def update_GPU_resources(run_config_input, RESERVED_GPU=0.05):

    available_GPUs = torch.cuda.device_count() #1 # ray.get_gpu_ids not working within slurm len(ray.get_gpu_ids())
    print(f'RAY available GPUs {available_GPUs}')

    if available_GPUs == 0:
        run_config_input["config"]["num_gpus"] = 0
        return run_config_input
    else:
        number_of_experiments = check_for_total_number_of_experiments(run_config_input)
        usable_GPU = available_GPUs-RESERVED_GPU

        worker_GPU = usable_GPU/number_of_experiments
        run_config_input["config"]["num_gpus"] = float(f'{worker_GPU:.4f}')
        return run_config_input

    #run_config_readable = update_experiments_for_GPU_resources(run_config_readable, available_GPUs=available_GPUs)

def check_for_total_number_of_experiments(run_config_input):
    list_grid_search_sizes = check_dict_for_grid_search(dict_to_scan=run_config_input)
    amount_experiments = 1
    for grid_size in list_grid_search_sizes:
        amount_experiments*=grid_size
    #potentially also include number of samples if we do not use grid_saerch
    return amount_experiments

def check_dict_for_grid_search(dict_to_scan, list_grid_search_sizes=[]):
    for key, value in dict_to_scan.items():
        if key == "grid_search":
            #value is a list of values
            list_grid_search_sizes.append(len(value))
            break
        if isinstance(value, dict):
            check_dict_for_grid_search(value, list_grid_search_sizes)
    return list_grid_search_sizes


def check_for_custom_eval_function(run_config_input):

    if "custom_eval_function" in run_config_input.get("config"):
        if run_config_input.get("config").get("custom_eval_function") == "custom_evaluation_incl_quantile":
            from evaluation_function_old import custom_evaluation_incl_quantile
            run_config_input["config"]["custom_eval_function"] = custom_evaluation_incl_quantile
        elif run_config_input.get("config").get("custom_eval_function") == "custom_eval_function_non_risk":
            from evaluation_function_old import custom_eval_function_non_risk
            run_config_input["config"]["custom_eval_function"] = custom_eval_function_non_risk
        elif run_config_input.get("config").get("custom_eval_function") == "custom_eval_function_lambda_penalty":
            from evaluation_function import custom_eval_function_lambda_penalty
            run_config_input["config"]["custom_eval_function"] = custom_eval_function_lambda_penalty
        else:
            print(f"Unknown custom eval function")
    return run_config_input

def check_for_custom_callbacks(run_config_input):

    if "callbacks" in run_config_input.get("config"):
        if run_config_input.get("config").get("callbacks") == "EvaluationLoggerCallback":
            from custom_callback import EvaluationLoggerCallback
            run_config_input["config"]["callbacks"] = EvaluationLoggerCallback
            #api_key_file = "wand_api.txt"
            #run_config_input.get("config").pop("callbacks")
            #from ray.tune.integration.wandb import WandbLoggerCallback

            #run_config_input["callbacks"] = [WandbLoggerCallback(api_key_file=api_key_file, project="test")]
    return run_config_input

def check_for_non_config_callbacks(run_config_input, group_name=None, project_name="shortselling"):


    if "callbacks" in run_config_input:
        tmp_callback_list = run_config_input.get("callbacks")
        run_config_input["callbacks"] = []
        for callback_entry in tmp_callback_list:
            if callback_entry == "WandbLoggerCallback":
                from ray.tune.integration.wandb import WandbLoggerCallback
                api_key_file = "wand_api.txt"
                run_config_input["callbacks"].append(WandbLoggerCallback(api_key_file=api_key_file, project=project_name,
                                                                         group=group_name))

            elif callback_entry == "WandbLoggerCustomCallback":
                from custom_wandb_logger import WandbLoggerCustomCallback
                api_key_file = "wand_api.txt"
                run_config_input["callbacks"].append(
                    WandbLoggerCustomCallback(api_key_file=api_key_file, project=project_name,
                                              group=group_name))
    return run_config_input

def check_for_constraint_amount_lambda_model(run_config_input):
    """
    Adds the amount of constraints to the model config in case we have a RiskLambdaCustomModel
    :param run_config_input:
    :return:
    """

    if "custom_model" in run_config_input.get("config").get("model") and \
            run_config_input.get("config").get("model").get("custom_model") == "RiskLambdaCustomModel" and \
            "head_factor_list" in run_config_input.get("config").get("env_config"):
        constraints_conditional_minkowski_encoding_type = run_config_input.get("config").get("env_config").get(
            "constraints_conditional_minkowski_encoding_type", None)
        if constraints_conditional_minkowski_encoding_type == "S4_U1":
            run_config_input.get("config").get("model").get("custom_model_config").get("config_lambda_model")[
                "amount_constraints"] = 2  # this is the special case for n2
            run_config_input.get("config").get("model").get("custom_model_config").get("config_lambda_model")[
                "constraints_conditional_minkowski_encoding_type"] = constraints_conditional_minkowski_encoding_type
        else: #standard case
            run_config_input.get("config").get("model").get("custom_model_config").get("config_lambda_model")["amount_constraints"] =\
                len(run_config_input.get("config").get("env_config").get("head_factor_list"))
    if "custom_model" in run_config_input.get("config").get("model") and \
            run_config_input.get("config").get("model").get("custom_model") == "CostValueFunctionCustomModel" and \
            "head_factor_list" in run_config_input.get("config").get("env_config"):
        constraints_conditional_minkowski_encoding_type = run_config_input.get("config").get("env_config").get("constraints_conditional_minkowski_encoding_type", None)
        if constraints_conditional_minkowski_encoding_type=="S4_U1":
            run_config_input.get("config").get("model").get("custom_model_config")[
                "amount_constraints"] = 2 #this is the special case for n2
            run_config_input.get("config").get("model").get("custom_model_config")["constraints_conditional_minkowski_encoding_type"] = constraints_conditional_minkowski_encoding_type
        else: #standard case
            run_config_input.get("config").get("model").get("custom_model_config")["amount_constraints"] =\
                len(run_config_input.get("config").get("env_config").get("head_factor_list"))

    return run_config_input


def set_experiment_name(run_config_input, run_name):

    experiment_name = f"{run_name}_{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}"
    run_config_input["name"] = experiment_name
    return run_config_input

def check_for_env_config_id(run_config_raw):

    if "env_config" in run_config_raw.get("config"):
        if "env_config_id" in run_config_raw.get("config").get("env_config"):
            env_config_params = get_environment_config(run_config_raw.get("config").get("env_config").pop("env_config_id"))
            for key, value in env_config_params.items():
                run_config_raw.get("config").get("env_config")[key]= value

    return run_config_raw

def get_environment_config(env_config_id):
    yaml_file_name = "config_environment_ids.yaml"

    with open(f"run_config/{yaml_file_name}", "r") as stream:
        try:
            run_config_raw = next(iter(yaml.safe_load(stream).values()))
        except yaml.YAMLError as exc:
            print(exc)

    env_config_params = run_config_raw.get(f"environment_config_id_{str(env_config_id)}", None)

    if None:
        raise ValueError(f"unknown environment_config id: {str(env_config_id)}")

    return env_config_params