from ray.rllib.algorithms import ppo
from ray.rllib.env.env_context import EnvContext
import argparse
import os
#import yaml
import ruamel.yaml as yaml
import sys

print("Training automatically with Ray Tune")
#results = tune.run(args.run, config=config, stop=stop)

#tuner = tune.Tuner(
#    args.run, param_space=config, run_config=air.RunConfig(stop=stop)
#)
#results = tuner.fit()

#ray.shutdown()

import argparse
import os

from ray.rllib.utils.test_utils import check_learning_achieved
from helper_config import convert_yaml_into_readable_format, convert_runable_into_2_0_format, register_custom_env, \
    register_algorithm_from_config, register_custom_action_distribution, register_custom_model, update_GPU_resources, \
    set_experiment_name, check_for_non_config_callbacks, check_for_custom_callbacks, check_for_custom_eval_function, \
    check_for_constraint_amount_lambda_model


parser = argparse.ArgumentParser()
parser.add_argument(
    "--run", type=str, default="PPO", help="The RLlib-registered algorithm to use."
)
parser.add_argument("--num-cpus", type=int, default=0)
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "tfe", "torch"],
    default="torch",
    help="The DL framework specifier.",
)
parser.add_argument("--eager-tracing", action="store_true")
parser.add_argument("--use-prev-action", action="store_true")
parser.add_argument("--use-prev-reward", action="store_true")
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.",
)
parser.add_argument(
    "--stop-iters", type=int, default=200, help="Number of iterations to train."
)
parser.add_argument(
    "--stop-timesteps", type=int, default=100000, help="Number of timesteps to train."
)
parser.add_argument(
    "--stop-reward", type=float, default=150.0, help="Reward at which we stop training."
)

if __name__ == "__main__":
    import ray
    from ray import air, tune

    #args = parser.parse_args()

    ray.init()#num_cpus=args.num_cpus or None)
    """
    configs = {
        "PPO": {
            "num_sgd_iter": 5,
            "model": {
                "vf_share_layers": True,
            },
            "vf_loss_coeff": 0.0001,
        },
        "IMPALA": {
            "num_workers": 2,
            "num_gpus": 0,
            "vf_loss_coeff": 0.01,
        },
    }

    config = dict(
        configs[args.run],
        **{
            "env": StatelessCartPole,
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            "model": {
                "use_lstm": True,
                "lstm_cell_size": 256,
                "lstm_use_prev_action": args.use_prev_action,
                "lstm_use_prev_reward": args.use_prev_reward,
            },
            "framework": args.framework,
            # Run with tracing enabled for tfe/tf2?
            "eager_tracing": args.eager_tracing,
        }
    )

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    # To run the Algorithm without ``Tuner.fit``, using our LSTM model and
    # manual state-in handling, do the following:

    # Example (use `config` from the above code):
    # >> import numpy as np
    # >> from ray.rllib.algorithms.ppo import PPO
    # >>
    # >> algo = PPO(config)
    # >> lstm_cell_size = config["model"]["lstm_cell_size"]
    # >> env = StatelessCartPole()
    # >> obs = env.reset()
    # >>
    # >> # range(2) b/c h- and c-states of the LSTM.
    # >> init_state = state = [
    # ..     np.zeros([lstm_cell_size], np.float32) for _ in range(2)
    # .. ]
    # >> prev_a = 0
    # >> prev_r = 0.0
    # >>
    # >> while True:
    # >>     a, state_out, _ = algo.compute_single_action(
    # ..         obs, state, prev_a, prev_r)
    # >>     obs, reward, done, _ = env.step(a)
    # >>     if done:
    # >>         obs = env.reset()
    # >>         state = init_state
    # >>         prev_a = 0
    # >>         prev_r = 0.0
    # >>     else:
    # >>         state = state_out
    # >>         prev_a = a
    # >>         prev_r = reward
    """ or None
    #print(config)

    if len(sys.argv) > 1:
        yaml_file_name = sys.argv[1]
    elif os.environ.get("YAML_FILE_NAME", None) is not None:
        yaml_file_name = os.environ.get("YAML_FILE_NAME", None)
    else:
        # yaml_file_name = "finance-ppo.yaml" # "custom-gridworld-riskppo.yaml"#"cartpole-ppo-test.yaml"  # "cartpole-ppo-test.yaml"
        # yaml_file_name = "finance-ppo-12.yaml"
        # yaml_file_name = "finance-ppo-2-working-12-12-21.yaml"
        # yaml_file_name = "finance-ppo-2-one-state.yaml"
        # yaml_file_name = "finance-ppo-penalty-state.yaml"
        #yaml_file_name = "finance-ppo-38.yaml"#"finance-ppo-36.yaml"#"finance-ppo-34.yaml"
        #yaml_file_name = "finance-ppo-36.yaml"
        #yaml_file_name = "cartpole-ddpg.yaml"

        #yaml_file_name = "finance-ddpg-4.yaml"
        #yaml_file_name = "finance-ppo-43.yaml"
        #yaml_file_name = "finance-ppo-47.yaml"
        #yaml_file_name = "finance-ppo-2-interval.yaml"
        #yaml_file_name = "pendulum-td3.yaml"
        #yaml_file_name = "pendulum-td3-custom.yaml"
        #yaml_file_name = "finance-td3-1.yaml"
        #yaml_file_name = "finance-ppo-57.yaml"
        #yaml_file_name = "finance-ppo-classic-1.yaml"
        #yaml_file_name = "finance-ddpg-1-interval.yaml"
        #yaml_file_name = "short-ppo-38.yaml"#"short-ppo-30.yaml"#"short-ppo-15.yaml"#"short-ppo-2.yaml"#"short-ppo-2.yaml"
        #yaml_file_name = "cartpole-stateless-lstm-2.yaml"
        #yaml_file_name = "short-ppo-155.yaml"#"short-ppo-141.yaml"#"short-ppo-101.yaml"#"short-ppo-90.yaml"#"short-ppo-85.yaml"#"short-ppo-85.yaml" #"short-ppo-69.yaml"#"short-ppo-53.yaml"#"short-ppo-51.yaml"#"short-ppo-49.yaml"#"short-ppo-44.yaml" #41 is working with custom_dirichlet distribution
        #yaml_file_name = "cartpole-stateless-lstm-1.yaml"
        #yaml_file_name = "ppo-autoreg-test-4.yaml"
        #yaml_file_name = "ppo-autoreg-test-6.yaml"
        #yaml_file_name = "ppo-autoreg-test-9.yaml"
        #yaml_file_name = "ppo-autoreg-test-12.yaml"
        #yaml_file_name = "ppo-autoreg-test-13.yaml"
        #yaml_file_name = "ppo-autoreg-const-20.yaml"
        #yaml_file_name = "ppo-autoreg-const-37.yaml"
        #yaml_file_name = "ppo-autoreg-const-40.yaml"
        #yaml_file_name = "ppo-autoreg-const-54.yaml"
        yaml_file_name = "ppo-autoreg-const-372.yaml"#"ppo-autoreg-const-354.yaml"#"ppo-autoreg-const-344.yaml"#"ppo-autoreg-const-155.yaml"

    print("RAY INIT")

    #yaml_file_name = "cartpole-stateless-framestacking.yaml"
    #yaml_file_name = "cartpole-stateless-lstm.yaml"
    #yaml_file_name = "cartpole-stateless-attention.yaml"


    with open(f"run_config/{yaml_file_name}", "r") as stream:
        try:
            run_config_raw = next(iter(yaml.safe_load(stream).values()))
        except yaml.YAMLError as exc:
            print(exc)

    run_name = yaml_file_name.replace(".yaml", "")

    PROJECT_NAME = "const_autoreg_2"
    print(run_config_raw)

    run_config_readable = convert_yaml_into_readable_format(run_config_input=run_config_raw)
    run_config_readable = register_custom_env(run_config_readable)
    run_config_readable = register_custom_action_distribution(run_config_readable)
    run_config_readable = register_algorithm_from_config(run_config_readable)
    run_config_readable = register_custom_model(run_config_readable)
    run_config_readable = check_for_constraint_amount_lambda_model(run_config_readable)
    run_config_readable = update_GPU_resources(run_config_readable)
    run_config_readable = check_for_custom_eval_function(run_config_readable)
    run_config_readable = check_for_custom_callbacks(run_config_readable)
    run_config_readable = check_for_non_config_callbacks(run_config_readable, group_name=run_name, project_name=PROJECT_NAME)
    run_config_readable = set_experiment_name(run_config_readable, run_name=run_name)

    print(run_config_readable)
    #must be last
    trainable, dict_param_space, dict_run_config = convert_runable_into_2_0_format(run_config_readable)

    #print(dict_run_config)

    #tuner = tune.Tuner(
    #    trainable=args.run, param_space=config, run_config=air.RunConfig(stop=stop, verbose=2)
    #)

    tuner = tune.Tuner(
        trainable=trainable, param_space=dict_param_space, run_config=air.RunConfig(**dict_run_config)
    )

    results = tuner.fit()

    ray.shutdown()