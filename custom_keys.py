"""
class Metric_Logging_Custom():
    SQUARED_LOSS_FIRST_MOMENT = "squared_loss_first_moment"
    SQUARED_LOSS_SECOND_MOMENT = "squared_loss_second_moment"
    ESTIMATED_FIRST_MOMENT_PER_TIME_STEP = "estimated_first_moment_per_time_step"
    ESTIMATED_SECOND_MOMENT_PER_TIME_STEP = "estimated_second_moment_per_time_step"


class Result_Keys_Custom():
    CURRENT_POLICY_ACTIONS = "current_policy_actions"
    CURRENT_POLICY_ACTIONS_EST_STD = "current_policy_actions_estimated_std"
""" or None

class Postprocessing_Custom():
    #RISK
    SQUARED_REWARDS_BASE_WITHOUT_RISK_PENALTY = "squared_rewards_base_without_risk_penalty"
    REWARDS_INCL_RISK = "rewards_incl_risk_penalty"
    REWARDS_BASE_WITHOUT_RISK_PENALTY = "rewards_base_without_risk_penalty"
    #ADVANTAGES_INCL_RISK = "advantages_incl_risk"
    #VALUE_TARGET_INCL_RISK = "value_target_incl_risk"
    #DISCOUNTED_CUMULATIVE_REWARDS = "discounted_cumulative_rewards"
    #MASK_BEGIN_TRAJECTORY = "mask_begin_trajectory"

    RISK_PENALTY_PARAMETER_VALUE = "risk_penalty_parameter_value"
    RISK_PENALTY = "risk_penalty_only"
    RISK_VARIANCE_ESTIMATE = "risk_variance_estimate"
    RISK_MVPI_FORMULATION = "risk_mvpi_formulation"
    MSE_FIRST_MOMENT = "mse_first_moment"
    MSE_SECOND_MOMENT = "mse_second_moment"
    DEBUG_MOMENTS = "debug_moment" #FIXME To be deleted later
    REWARDS_DETAILS = "rewards_details"
    SEQ_LENS_DUMMY = "seq_len_dummy"
    ALLOCATION = "allocation" #just used as tmp
    ALLOCATION_mean = "allocation_mean"
    ALLOCATION_min = "allocation_min"
    ALLOCATION_max = "allocation_max"
    AMOUNT_ALLOCATION_VIOLATIONS = "amount_allocation_violations"
    REWARD_BASE_NO_PENALTIES = "rewards_base_no_penalties"
    LAMBDA_WEIGHTED_CONSTRAINT_PENALTIES = "lambda_weighted_constraint_penalties"
    UNWEIGHTED_PENALTY_VIOLATION_SCORE = "unweighted_penalty_violation_score"
    REWARD_INCL_CONSTRAINT_PENALTIES = "rewards_incl_constraint_penalties"
    LOG_BARRIER_CONSTRAINT_PENALTIES = "log_barrier_constraint_penalties"
    COST_ADVANTAGES = "cost_advantages_"
    COST_VALUE_TARGETS = "cost_value_targets_"

    AMOUNT_CONSTRAINT_VIOLATIONS = "amount_constraint_violations"

    BOOL_ANY_CONSTRAINT_VIOLATIONS = "bool_any_constraint_violations"

    ACTION_VIOLATIONS = "action_violation_samples"
    ACTION_VIOLATIONS_CONSTRAINT_VIOLATIONS = "action_violation_constraint_index_samples"

    PROCESSED_ACTIONS = "processed_actions"

class SampleBatch_Custom():
    COST_VF_PREDS = "cost_vf_pred_" #this is used in combination with index numbers, i.e. f'{COST_VF_PREDS_}{idx}'
    COST_OBS = "cost_obs_"
"""
class SampleBatch_Custom():
    ACTION_STATE_REWARD_FIRST_MOMENT_ESTIMATE = "action_state_reward_first_moment_estimate"
    ACTION_STATE_REWARD_SECOND_MOMENT_ESTIMATE = "action_state_reward_second_moment_estimate"
    ACTION_STATE_REWARD_STANDARD_DEVIATION_ESTIMATE = "action_state_reward_standard_deviation_estimate"
    VF_PREDS_INCL_RISK = "value_function_prediction_incl_risk"
    AGENT_RISK_TOLERANCE_LEVEL_IN_STANDARD_DEVIATIONS = "agent_risk_tolerance_in_standard_deviations"
""" or None