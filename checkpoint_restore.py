import sys
import os
import re
import json


from helper_config import convert_yaml_into_readable_format, convert_runable_into_2_0_format, register_custom_env, \
    register_algorithm_from_config, register_custom_action_distribution, register_custom_model, update_GPU_resources, \
    set_experiment_name, check_for_non_config_callbacks, check_for_custom_callbacks, check_for_custom_eval_function, \
    check_for_constraint_amount_lambda_model


def find_correct_checkpoint_folder_file(experiment_path, checkpoint_id):
    list_checkpoints = [f for f in os.listdir(experiment_path) if re.match(f'checkpoint_{int(checkpoint_id):06}', f)] #6digits with leading 0
    return f'{experiment_path}/{list_checkpoints[0]}/checkpoint-{checkpoint_id}'

def identify_experiment_path(experiment_raw_path):
    root_path = f".logging/"
    list_experiment_path = [f for f in os.listdir(root_path) if re.match(f'^(?!.*eval){experiment_raw_path}.*', f)]
    if len(list_experiment_path)>1:
        raise ValueError(f"More than one possible experiments:{list_experiment_path}")
    return list_experiment_path[0]

def replace_config_strings(dict_params):

    dict_string_mappings = {
        'callbacks': ['EvaluationLoggerCallback'],
        'custom_eval_function': ['custom_eval_function_lambda_penalty']
    }

    for key, val in dict_string_mappings.items():
        for entry in val:
            if re.match(f".*{entry}.*", dict_params.get(key)):
                dict_params[key] = entry
                break

    return dict_params

def convert_experiments_params_into_runnable_config(experiment_path, experiment_id=None,
                                                    checkpoint_id=None):
    if experiment_path is None:
        root_path = ".logging/short-ppo-145_14-10-2022_10-32-27"  # "RiskPPO_2022-03-29_09-00-12"
    else:
        root_path = f".logging/{experiment_path}"

    list_experiments = [f for f in os.listdir(root_path) if re.match(f'{experiment_id}.*', f)]
    experiment_path = f'{root_path}/{list_experiments[0]}'

    # retriving param file:
    #print(experiment_path)
    # try:
    with open(f'{experiment_path}/params.json') as f:
        dict_config = json.load(f)
    dict_config = replace_config_strings(dict_config)

    full_dict = {}
    # MAKE SURE TO USE THE CORRECT RUN + custom_eval_function + callbacks
    full_dict["config"] = dict_config
    full_dict["config"]["env"] = full_dict["config"]["env"].replace("wrapped-", "")
    full_dict["config"]["evaluation_interval"] = int(checkpoint_id) + 1

    #removing seed setting since this causes problems with gym restore
    #full_dict["config"].pop("seed")

    full_dict["run_or_experiment"] = experiment_id.split("_")[0]  ##replacement_dict.get("trainer_type")

    full_dict["stop"] = {}
    full_dict["stop"]["training_iteration"] = int(checkpoint_id) + 1
    full_dict["local_dir"] = "./.logging"

    # ray 2.0 formal
    #full_dict["checkpoint_config"] = {}
    #full_dict["checkpoint_config"]["checkpoint_at_end"] = False
    full_dict["checkpoint_at_end"] = False

    full_dict["verbose"] = 1

    full_dict["callbacks"] = ["WandbLoggerCallback"]

    full_dict["restore"] = find_correct_checkpoint_folder_file(experiment_path=experiment_path, checkpoint_id=checkpoint_id)

    #amount of episodes to evaluate
    # Amount of episodes for evaluation
    full_dict["config"]["evaluation_amount_episodes"] = 1000
    full_dict["config"]["evaluation_allow_backtesting"] = True

    return full_dict, experiment_path

if __name__ == "__main__":
    import ray
    from ray import air, tune

    # args = parser.parse_args()

    # README : RUN ONLY ON THE MACHINE YOU TRAINED THE MODEL, if you train on slurm then it wont work locally
    # python checkpoint_restore.py short-ppo-156_15-10-2022_10-01-37 RiskShortPPO_wrapped-financial-markets-env-short-selling-v0_985e9_00000 1
    # python checkpoint_restore.py short-ppo-156 RiskShortPPO_wrapped-financial-markets-env-short-selling-v0_985e9_00000 1

    ray.init()  # num_cpus=args.num_cpus or None)

    experiment_id = None
    sub_experiment_id = None
    checkpoint_id = None
    if len(sys.argv) > 1:
        #experiment_id = sys.argv[1]
        experiment_raw_path = sys.argv[1] # this is not the full name
        experiment_path = identify_experiment_path(experiment_raw_path=experiment_raw_path)
    if len(sys.argv) > 2:
        #sub_experiment_id = sys.argv[2]
        experiment_id = sys.argv[2]
    if len(sys.argv) > 3:
        checkpoint_id = sys.argv[3]

    if experiment_path is None:
        experiment_path = "short-ppo-156_15-10-2022_10-01-37"#"short-ppo-145_14-10-2022_10-32-27"
        experiment_id = "RiskShortPPO_wrapped-financial-markets-env-short-selling-v0_985e9_00000"
        checkpoint_id = 1#500

    evaluation_run_name = f'{experiment_path.split("_")[0]}_eval'

    run_config_raw, experiment_full_path = convert_experiments_params_into_runnable_config(experiment_path=experiment_path,
                                                    experiment_id=experiment_id,
                                                    checkpoint_id=checkpoint_id)

    run_name = evaluation_run_name #yaml_file_name.replace(".yaml", "")

    run_config_readable = convert_yaml_into_readable_format(run_config_input=run_config_raw)
    run_config_readable = register_custom_env(run_config_readable)
    run_config_readable = register_custom_action_distribution(run_config_readable)
    run_config_readable = register_algorithm_from_config(run_config_readable)
    run_config_readable = register_custom_model(run_config_readable)
    run_config_readable = check_for_constraint_amount_lambda_model(run_config_readable)
    run_config_readable = update_GPU_resources(run_config_readable)
    run_config_readable = check_for_custom_eval_function(run_config_readable)
    run_config_readable = check_for_custom_callbacks(run_config_readable)
    run_config_readable = check_for_non_config_callbacks(run_config_readable, group_name=run_name, project_name="const_autoreg")
    run_config_readable = set_experiment_name(run_config_readable, run_name=run_name)

    print(run_config_readable)

    #must be last
    # BEG:: This would be 2.0 notation
    #trainable, dict_param_space, dict_run_config = convert_runable_into_2_0_format(run_config_readable)

    #tuner = tune.Tuner(
    #    trainable=trainable, param_space=dict_param_space, run_config=air.RunConfig(**dict_run_config)
    #)
    #results = tuner.fit()
    # or simply get the last checkpoint (with highest "training_iteration")
    #last_checkpoint = ray.tune.analysis.get_last_checkpoint()
    # if there are multiple trials, select a specific trial or automatically
    # choose the best one according to a given metric
    #last_checkpoint = ray.tune.analysis.get_last_checkpoint(
    #    metric="episode_reward_mean", mode="max"
    #)
    #tuner = tuner.restore(path=find_correct_checkpoint_folder_file(experiment_path=experiment_full_path, checkpoint_id=checkpoint_id))
    #results = tuner.fit()
    #tuner = tune.Tuner.restore("/home/winkel/GoogleDrive/Documents/PhD/code/rl_rllib_2_0_risk_short_selling/.logging/short-ppo-145_14-10-2022_10-32-27/RiskShortPPO_wrapped-financial-markets-env-short-selling-v0_bc9aa_00000_0_lr=0.0000,fcnet_hiddens=512_256_128_2022-10-14_10-32-28/checkpoint_000500/checkpoint-500").fit()
    #tuner = tuner.restore("short-ppo-145_14-10-2022_10-32-27").fit()
    #END:: This would be 2.0 notation


    #We use the old logic, not the 2.0 notation
    #path = "/home/winkel/GoogleDrive/Documents/PhD/code/rl_rllib_2_0_risk_short_selling/.logging/short-ppo-145_14-10-2022_10-32-27/RiskShortPPO_wrapped-financial-markets-env-short-selling-v0_bc9aa_00000_0_lr=0.0000,fcnet_hiddens=512_256_128_2022-10-14_10-32-28/checkpoint_000500"
    #config = run_config_readable["config"]

    #print(config)

    #WORKING
    results = tune.run(**run_config_readable)

    """
    from risk_short_ppo_algorithm import RiskShortPPO
    results = tune.run(
        RiskShortPPO,  # RiskIPOMergedPPO,
        # stop=stop,
        config=config,
        restore=path,  # best_checkpoint.to_directory(),
        verbose=1,
    )
    """ or None
    #"/home/winkel/GoogleDrive/Documents/PhD/code/rl_rllib_2_0_risk_short_selling/.logging/short-ppo-155_15-10-2022_08-26-50/RiskIPOMergedPPO_wrapped-financial-markets-env-short-selling-v0_5a4ec_00000_0_2022-10-15_08-26-50/checkpoint_000001"

    ray.shutdown()
