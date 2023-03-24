import torch
import numpy as np
from typing import List
import pypoman
from scipy.optimize import linprog
from mhar import walk
import polytope


class RELATIONTYPE():
    EQ = "=="
    LE = "<="
    GE = ">="

class PolytopeEvaluator():

    def __init__(self, list_full_agg_constraint_tuples, batch_size=1):
        self.number_groups = len(list_full_agg_constraint_tuples[0][0])
        self.batch_size = batch_size
        self.original_list_full_agg_constraint_tuples = list_full_agg_constraint_tuples
        self.reset_polytope_evaluator()

        self.availabe_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #must be able to batch process
        self.sample_size_mhar_package = 200
        self.sample_size_polytope_package = 200


    def reset_polytope_evaluator(self):
        self.list_list_full_agg_constraint_tuples = [self.original_list_full_agg_constraint_tuples.copy() for _ in
                                                     range(self.batch_size)]
        self.list_list_all_samples = [[None] * self.number_groups for _ in range(self.batch_size)]
        self.list_last_valid_vertices_list = [None for _ in range(self.batch_size)]

        self.update_current_upper_lower_bounds()

    def update_current_upper_lower_bounds(self):
        self.list_current_lower_bounds, self.list_current_upper_bounds = self.calculate_upper_lower_bound_for_all_indices()

    def get_current_upper_lower_bounds_for_index(self, index):

        return np.array(self.list_current_lower_bounds)[:,index].tolist(), np.array(self.list_current_upper_bounds)[:,index].tolist()

    @staticmethod
    def convert_dict_polytope_in_list_full_agg_constraint_tuples(dict_polytope):
        amount_constraints = len(dict_polytope.get("head_factor_list"))
        amount_groups = len(dict_polytope.get("action_mask_dict").get("0_action_mask"))
        list_full_agg_constraint_tuple = []

        tmp_list_mask = [1] * amount_groups
        list_full_agg_constraint_tuple.append((tmp_list_mask, 1.0, '>='))
        list_full_agg_constraint_tuple.append((tmp_list_mask, 1.0, '<='))

        for i in range(amount_constraints):
            list_full_agg_constraint_tuple.append((dict_polytope.get("action_mask_dict").get(f"{i}_action_mask"),
                                                   dict_polytope.get("head_factor_list")[i], '>='))

        for j in range(amount_groups):
            tmp_list_mask = [0] * amount_groups
            tmp_list_mask[j] = 1
            list_full_agg_constraint_tuple.append((tmp_list_mask, 0.0, '>='))

        return list_full_agg_constraint_tuple

    def update_list_list_full_agg_constraint_tuples_by_index_sample(self, list_interval_samples, index):

        for ctx, sample in enumerate(list_interval_samples):
            tmp_list_mask = [0] * self.number_groups
            tmp_list_mask[index] = 1
            #which is equivalent to adding a == constraint to the value on index
            self.list_list_full_agg_constraint_tuples[ctx].append((tmp_list_mask, sample, '>='))
            self.list_list_full_agg_constraint_tuples[ctx].append((tmp_list_mask, sample, '<='))
            self.list_list_all_samples[ctx][index]=sample
            #print(self.list_list_all_samples)

        self.update_current_upper_lower_bounds()

    @DeprecationWarning
    def calculate_upper_lower_bound_for_index(self, index)->List:

        list_min_val_index = []
        list_max_val_index = []

        for batch_index, list_full_agg_constraint_tuple in enumerate(self.list_list_full_agg_constraint_tuples):
            np_lhs, np_rhs, list_relation_types = PolytopeEvaluator.convert_masked_agg_tuple_list_into_LE_system(
                list_tuple_agg_constraints=list_full_agg_constraint_tuple)
            np_lhs, np_rhs, list_relation_types = PolytopeEvaluator.convert_system_in_all_LE_constraints(np_lhs=np_lhs,
                                                                                       np_rhs=np_rhs,
                                                                                       list_relation_types=list_relation_types)
            try:
                # this is necessary due to numerical problems with pypoman where the package sometimes encounters
                # numerical problems
                vertices_orig = pypoman.compute_polytope_vertices(np_lhs, np_rhs)
            except:
                vertices_orig = self.list_last_valid_vertices_list[batch_index]

            if len(vertices_orig)>0:
                self.list_last_valid_vertices_list[batch_index] = vertices_orig
            else:
                vertices_orig = self.list_last_valid_vertices_list[batch_index]

            np_vertices_orig = np.vstack(vertices_orig)
            np_max_val = np.max(np_vertices_orig, axis=0)
            np_min_val = np.min(np_vertices_orig, axis=0)

            list_min_val_index.append(np_min_val[index])
            list_max_val_index.append(np_max_val[index])

        return list_min_val_index, list_max_val_index


    def calculate_upper_lower_bound_for_all_indices(self)->List:

        list_min_val_index = []
        list_max_val_index = []

        for batch_index, list_full_agg_constraint_tuple in enumerate(self.list_list_full_agg_constraint_tuples):
            np_lhs, np_rhs, list_relation_types = PolytopeEvaluator.convert_masked_agg_tuple_list_into_LE_system(
                list_tuple_agg_constraints=list_full_agg_constraint_tuple)
            np_lhs, np_rhs, list_relation_types = PolytopeEvaluator.convert_system_in_all_LE_constraints(np_lhs=np_lhs,
                                                                                       np_rhs=np_rhs,
                                                                                       list_relation_types=list_relation_types)
            try:
                # this is necessary due to numerical problems with pypoman where the package sometimes encounters
                # numerical problems
                vertices_orig = pypoman.compute_polytope_vertices(np_lhs, np_rhs)
            except:
                vertices_orig = self.list_last_valid_vertices_list[batch_index]

            if len(vertices_orig)>0:
                self.list_last_valid_vertices_list[batch_index] = vertices_orig
            else:
                vertices_orig = self.list_last_valid_vertices_list[batch_index]

            np_vertices_orig = np.vstack(vertices_orig)
            np_max_val = np.max(np_vertices_orig, axis=0)
            np_min_val = np.min(np_vertices_orig, axis=0)

            list_min_val_index.append(np_min_val)
            list_max_val_index.append(np_max_val)

        return list_min_val_index, list_max_val_index


    @staticmethod
    def convert_masked_agg_tuple_list_into_LE_system(list_tuple_agg_constraints):
        """
        System is masked since the positions are either 0 or 1
        :param list_tuple_agg_constraints:
        :return:
        """
        list_lhs_system = []
        list_rhs_system = []
        list_relation_types = []
        for tuple_agg_constraints in list_tuple_agg_constraints:
            equation_system = tuple_agg_constraints[0]
            rhs_value = tuple_agg_constraints[1]
            relationship_type = tuple_agg_constraints[2]
            list_lhs_system.append(equation_system)
            list_rhs_system.append([rhs_value])
            if relationship_type == RELATIONTYPE.EQ or relationship_type == "==":
                list_relation_types.append(RELATIONTYPE.EQ)
            elif relationship_type == RELATIONTYPE.LE or relationship_type == "<=":
                list_relation_types.append(RELATIONTYPE.LE)
            elif relationship_type == RELATIONTYPE.GE or relationship_type == ">=":
                list_relation_types.append(RELATIONTYPE.GE)

        np_lhs_system = np.array(list_lhs_system)
        np_rhs_system = np.array(list_rhs_system)
        return np_lhs_system, np_rhs_system, list_relation_types

    @staticmethod
    def convert_system_in_all_LE_constraints(np_lhs, np_rhs, list_relation_types):
        """
        np_lhs = A
        np_rhs = b
        convert to -> A->A' and b->b' with all <= relations
        A'x <= b'
        :param np_lhs:
        :param np_rhs:
        :param list_equations:
        :return:
        """

        list_new_relation_types = []
        list_new_lhs = []
        list_new_rhs = []
        for idx, row in enumerate(np_lhs):
            expression_type = list_relation_types[idx]
            rhs_value = np_rhs[idx][0]
            if expression_type == RELATIONTYPE.EQ:
                # pos values
                list_new_lhs.append(row)
                list_new_rhs.append(rhs_value)
                list_new_relation_types.append(RELATIONTYPE.LE)
                # neg values
                list_new_lhs.append(row * (-1))
                list_new_rhs.append(rhs_value * (-1))
                list_new_relation_types.append(RELATIONTYPE.LE)
            elif expression_type == RELATIONTYPE.LE:
                list_new_lhs.append(row)
                list_new_rhs.append(rhs_value)
                list_new_relation_types.append(RELATIONTYPE.LE)
            elif expression_type == RELATIONTYPE.GE:
                list_new_lhs.append(row * (-1))
                list_new_rhs.append(rhs_value * (-1))
                list_new_relation_types.append(RELATIONTYPE.LE)
            else:
                raise ValueError(f'Unknown')

        np_lhs_new = np.array(list_new_lhs)
        np_rhs_new = np.expand_dims(np.array(list_new_rhs), axis=1)

        return np_lhs_new, np_rhs_new, list_new_relation_types

    @staticmethod
    def sample_scaled_dirichlet(scaling_factor: float, number_variables, number_samples, list_mask):
        dirich_param = np.ones(number_variables)
        dirich_param_masked = PolytopeEvaluator.apply_mask(dirich_param, list_mask)

        dir_samples = np.random.dirichlet(dirich_param_masked, size=number_samples)
        #print(number_samples)
        return dir_samples * scaling_factor

    @staticmethod
    def apply_mask(np_ones, list_mask):
        a_min = 0.00000001 #needs to be sufficiently small
        return np_ones*np.clip(np.array(list_mask), a_min=a_min, a_max=1.0)

    def estimate_uniform_marginal_normalized_population_parameter_set_by_method(self, column_index, method='polytope_package'):
        if method == 'polytope_package':
            mean_input_array, variance_input_array = \
                self.estimate_uniform_marginal_normalized_population_parameter_set_hit_and_run(column_index=column_index)
            return PolytopeEvaluator.replace_nans_in_mean_variance_array(
                mean_input_array=mean_input_array,
                variance_input_array=variance_input_array,
                number_groups=self.number_groups,
                group_index=column_index
                )
        elif method == 'mhar_package':
            mean_input_array, variance_input_array = \
                self.estimate_uniform_marginal_normalized_population_parameter_set_hit_and_run_torch(column_index=column_index)
            return PolytopeEvaluator.replace_nans_in_mean_variance_array(
                mean_input_array=mean_input_array,
                variance_input_array=variance_input_array,
                number_groups=self.number_groups,
                group_index=column_index
            )
        elif method == 'static_estimate':
            return PolytopeEvaluator.get_static_estimate(number_groups=self.number_groups,
                                                         group_index=column_index,
                                                         batch_size=self.batch_size)
        else:
            raise ValueError(f'Unknown method')

    def estimate_uniform_marginal_normalized_population_parameter_set(self, column_index, list_min_val_index, list_max_val_index, debug_non_normalized=False):
        """
        This returns the parameters for NORMALIZED samples, i.e. we can sample from [0,1], only afterwards the
        real boundaries given by the polytope are incorporated using a linear transformation X_trans=a*x+b
        The actor however generates actions [0,1] (Which are then transformed in the environment)
        :param column_index:
        :param list_min_val_index:
        :param list_max_val_index:
        :return:
        """
        np_sampled = np.array(self.list_list_all_samples)
        number_samples = 10_000

        np_already_sampled = np_sampled[:,:column_index].astype(float)
        number_free_variables = self.number_groups-column_index

        #print("Already Sampled")
        #print(np_already_sampled)

        list_normalized_marginal_means_polytope = [] # list of means for each polytope (each row in np_already_sampled represents a (part) of a polytope
        list_normalized_marginal_variances_polytope = [] # list of variances for each polytope
        for sample_index, row in enumerate(np_already_sampled):
            #the last loop will always have zero variance, since everything is determined by then
            scaling_factor = np.sum(row)
            #print(f"ROW index {sample_index}")
            #print(row)

            tmp_duplicates = np.tile(row, (number_samples,1))
            samples = PolytopeEvaluator.sample_scaled_dirichlet(scaling_factor=(1-scaling_factor), number_variables=number_free_variables,
                                                                       number_samples=number_samples, list_mask=[1] * number_free_variables)
            tmp_merged_samples = np.concatenate((tmp_duplicates, samples), axis=1)

            #print(tmp_merged_samples)

            #t0 = time.time()
            np_penalty_matrix = PolytopeEvaluator.generate_penalty_matrix(tmp_merged_samples, self.list_list_full_agg_constraint_tuples[sample_index])

            #violation mask
            zero_matrix = np.zeros_like(np_penalty_matrix)
            tmp_bool_matrix = ~np.isclose(zero_matrix, np_penalty_matrix)
            any_violation_in_allocation = np.any(tmp_bool_matrix, axis=1)
            #t1 = time.time()
            #total = t1 - t0
            #print(f"We needed {total} time for penaltiy eval")

            #only take samples which are not in violation
            samples_non_violation = tmp_merged_samples[~any_violation_in_allocation]
            if samples_non_violation.size == 0:
                pass
                #print("§§§§§§§§§§")
                #print(self.list_list_full_agg_constraint_tuples[sample_index])
                #print("%%%%%")
                #print(row)
                #print(samples[any_violation_in_allocation])
                #print("%%%%%%%%%")
                #print(np_penalty_matrix[any_violation_in_allocation])
                #print("///////////")
                #print(scaling_factor)
                #print("WE have no samples in non-violation")
            else:
                #print("WE HAVE MATCHING")
                pass
            #selecting the current variable we need to estimate the population parameters vor
            np_relevant_variable_column = samples_non_violation[:,column_index]

            #print("RELEVANT COLUMN")
            #print(np_relevant_variable_column)

            #normalize the samples to a [0,1] interval (since we sample from that and then transform it back later) - this is polytope specific
            normalizing_scaling_factor = np.array(list_max_val_index[sample_index])-np.array(list_min_val_index[sample_index])
            normalizing_intercept_term = np.array(list_min_val_index[sample_index])
            #X_trans = a*X+b -> reverse engineer: X = (X_trans-b)/a
            #this might potentially produce inf/-inf if the normalizing_scaling_factor is zero.
            #normalizing_scaling_factor_broadcasted = np.empty(np_relevant_variable_column.shape).fill(normalizing_scaling_factor)
            #print(np_relevant_variable_column)
            #print(np.empty(np_relevant_variable_column.shape))
            #test = np.empty(np_relevant_variable_column.shape)
            #print(test)
            #print(test.fill(normalizing_scaling_factor))
            #print(np.empty(np_relevant_variable_column.shape).fill(normalizing_scaling_factor))
            #print("-----------")
            #np_relevant_variable_column_normalized = np.where(
            #    normalizing_scaling_factor_broadcasted<0.0000001,
            #    np.empty(np_relevant_variable_column.shape).fill(np.nan),
            #    (np_relevant_variable_column-np.ones_like(np_relevant_variable_column)*normalizing_intercept_term)/normalizing_scaling_factor_broadcasted)
            #if normalizing_scaling_factor < 0.0000001:
            #    np_relevant_variable_column_normalized = np.empty(np_relevant_variable_column.shape)
            #    np_relevant_variable_column_normalized.fill(np.nan)
            #else:
            #    np_relevant_variable_column_normalized = (np_relevant_variable_column-np.ones_like(np_relevant_variable_column)*normalizing_intercept_term)/normalizing_scaling_factor
            #print("LAST")
            #print(normalizing_scaling_factor)
            if normalizing_scaling_factor < 0.0005:
                normalizing_scaling_factor=0

            #print("Before normalization")
            #print(np_relevant_variable_column)
            np_relevant_variable_column_normalized = (np_relevant_variable_column - np.ones_like(
                np_relevant_variable_column) * normalizing_intercept_term) / normalizing_scaling_factor

            #if np.isnan(np_relevant_variable_column_normalized).any():
            #    list_normalized_marginal_means_polytope.append(np.nan)
            #    list_normalized_marginal_variances_polytope.append(np.nan)
            #else:
            if np_relevant_variable_column_normalized.size == 0:
                print("SLICE IS EMPTY")
                print(np_relevant_variable_column_normalized)

            if debug_non_normalized:
                list_normalized_marginal_means_polytope.append(np.mean(np_relevant_variable_column))
                list_normalized_marginal_variances_polytope.append(np.var(np_relevant_variable_column))
            else:
                list_normalized_marginal_means_polytope.append(np.mean(np_relevant_variable_column_normalized))
                list_normalized_marginal_variances_polytope.append(np.var(np_relevant_variable_column_normalized))
            #amount_violations = check_constraint_violations(np_penalty_matrix)

        #print(list_normalized_marginal_means_polytope)
        #print(list_normalized_marginal_variances_polytope)
        np_normalized_marginal_means_polytope = np.array(list_normalized_marginal_means_polytope)
        np_normalized_marginal_variances_polytope = np.array(list_normalized_marginal_variances_polytope)
        #Note the output here is all for NORMALIZED marginal means, i.e. [0,1], they are only converted back later
        return np_normalized_marginal_means_polytope, np_normalized_marginal_variances_polytope

    @staticmethod
    def generate_penalty_matrix(np_samples, list_agg_constraints, correct_close_to_error=True):

        list_penalty_vectors = []
        for agg_constraint in list_agg_constraints:
            tmp_mask = agg_constraint[0]
            constraint_val = agg_constraint[1]
            constraint_type = agg_constraint[2]
            amount_samples = np_samples.shape[0]

            tmp_mask = np.tile(tmp_mask, (amount_samples, 1))

            masked_samples = tmp_mask * np_samples
            sum_samples = np.sum(masked_samples, axis=1)  # we have the sum per sample
            constraint_vector = np.ones_like(sum_samples) * constraint_val
            delta_vector = sum_samples - constraint_vector

            if constraint_type == "<=":
                penalty_vector = np.maximum(0.0, delta_vector)
            if constraint_type == ">=":
                penalty_vector = np.minimum(0.0, delta_vector)

            abs_penalty_vector = np.abs(penalty_vector)
            if correct_close_to_error:
                zero_vector = np.zeros_like(abs_penalty_vector)
                tmp_bool_violation = np.isclose(zero_vector, abs_penalty_vector,
                                                atol=1e-06)  # deviations of less than 0.000001 are ignored
                abs_penalty_vector = np.where(tmp_bool_violation, 0.0, abs_penalty_vector)  # replaces close to
                # zero values by zero

            list_penalty_vectors.append(abs_penalty_vector)

        return np.transpose(np.array(list_penalty_vectors))

    @staticmethod
    def check_constraint_violations(np_penalty_matrix):
        """
        Returns total amount of solutions which have at least one constraint valuation
        :param list_constraints:
        :param list_penalty_vectors:
        :return:
        """
        zero_matrix = np.zeros_like(np_penalty_matrix)

        tmp_bool_matrix = ~np.isclose(zero_matrix, np_penalty_matrix)

        any_violation_in_allocation = np.any(tmp_bool_matrix, axis=1)
        # amount_violation_in_allocation = np.sum(tmp_bool_matrix, axis=1)
        amount_allocations_in_violation = np.sum(any_violation_in_allocation)

        return amount_allocations_in_violation

    @staticmethod
    def convert_to_distribution_parameters(distribution_type, np_sample_means, np_sample_variances):
        if distribution_type=="beta":
            alpha = np_sample_means*((np_sample_means-np_sample_means**2)/np_sample_variances-1)
            beta = alpha/np_sample_means-alpha
            return alpha, beta
        elif distribution_type=="dirichlet":
            alpha_1 = np_sample_means*((np_sample_means-np_sample_means**2)/np_sample_variances-1)
            alpha_2 = alpha_1/np_sample_means - alpha_1
            return alpha_1, alpha_2 # for n=2 same as for beta

    @staticmethod
    def transform_normalized_action_to_polytope_action(np_normalized_action,
                                                       list_min_val_for_index,
                                                       list_max_val_for_index) -> np.ndarray:
        if np_normalized_action.ndim == 2:

            #assert len(list_min_val_for_index) > 1
            #assert len(list_max_val_for_index) > 1

            np_lower_bounds = np.expand_dims(np.array(list_min_val_for_index), axis=1)
            np_upper_bounds = np.expand_dims(np.array(list_max_val_for_index), axis=1)
            # np_ones_dummy = np.ones_like(dummy_input)
            tmp_np_interval_width = np_upper_bounds - np_lower_bounds

            #np_scaling_val = dict_raw_actions.get(f"group_allocation_interval_{index_val}")
            np_allocation_value = np_lower_bounds + np_normalized_action * tmp_np_interval_width

            #list_transformed_allocation_value = np_allocation_value.flatten().tolist()
            return np_allocation_value #list_transformed_allocation_value
        elif np_normalized_action.ndim == 1:
            assert len(list_min_val_for_index) == 1
            assert len(list_max_val_for_index) == 1
            #np_lower_bounds = np.expand_dims(np.array(list_min_val_for_index), axis=1)
            #np_upper_bounds = np.expand_dims(np.array(list_max_val_for_index), axis=1)
            tmp_np_interval_width = (np.ones(1) * list_max_val_for_index[0]) - (np.ones(1) * list_min_val_for_index[0])

            np_allocation_value = (np.ones(1) * list_min_val_for_index[0]) + np_normalized_action * tmp_np_interval_width
            return np_allocation_value

    @staticmethod
    def get_static_estimate(number_groups, group_index, batch_size):
        mean_value, variance_value = PolytopeEvaluator.get_static_estimate_value_only(number_groups, group_index)
        return np.ones(batch_size)*mean_value, np.ones(batch_size)*variance_value

    @staticmethod
    def get_static_estimate_value_only(number_groups, group_index):
        grp_param_dict = {0: {'mean': 0.07696248855486437, 'variance': 0.005076243169097674},
                          1: {'mean': 0.08333226292989833, 'variance': 0.005878320466156978},
                          2: {'mean': 0.09090176016998894, 'variance': 0.006881862783234002},
                          3: {'mean': 0.09999091689030026, 'variance': 0.008183692815973647},
                          4: {'mean': 0.11111061484672136, 'variance': 0.009880684760614987},
                          5: {'mean': 1 / 8, 'variance': 0.01215249874405284},
                          6: {'mean': 0.1509604329004329, 'variance': 0.015305266818246474},
                          7: {'mean': 1 / 6, 'variance': 0.01983965567436485},
                          8: {'mean': 0.2, 'variance': 0.02666800433313894},
                          9: {'mean': 0.25, 'variance': 0.03754565801783012},
                          10: {'mean': 0.3333333333333333, 'variance': 0.05549660721059072},
                          11: {'mean': 0.5, 'variance': 0.08346961696521785},
                          12: {'mean': 0.99, 'variance': 0.0005}}
        index_to_use = (len(list(grp_param_dict.keys())) - number_groups) + group_index
        return grp_param_dict.get(index_to_use).get("mean"), grp_param_dict.get(index_to_use).get("variance")

    @staticmethod
    def enforce_low_variance_value(mean_input_array, variance_input_array):

        #floor value for variance, otherwise we run into Nans if we take the exponential later of the converted parameters
        # low variance which does not cause crashes yet 0.005
        return mean_input_array

    @staticmethod
    def replace_nans_in_mean_variance_array(mean_input_array, variance_input_array, number_groups, group_index):
        if isinstance(mean_input_array, torch.Tensor):
            torch_means = mean_input_array
            torch_variance = variance_input_array

            bool_mask_mean_nan_inf = ~torch.isfinite(mean_input_array)
            bool_mask_variance_nan_inf = ~torch.isfinite(variance_input_array)

            mean_value, variance_value = PolytopeEvaluator.get_static_estimate_value_only(number_groups, group_index)
            if bool_mask_mean_nan_inf.any():
                mean_input_array[bool_mask_mean_nan_inf] = mean_value

            if bool_mask_variance_nan_inf.any():
                variance_input_array[bool_mask_variance_nan_inf] = variance_value

            return mean_input_array, variance_input_array

            #print(torch_means)
            #print(torch_variance)
            #mean_value, variance_value = PolytopeEvaluator.get_static_estimate_value_only(number_groups, group_index)
            #param_1, param_2 = PolytopeEvaluator.convert_to_distribution_parameters(distribution_type, mean_value, variance_value)
            #raise NotImplementedError
        elif isinstance(mean_input_array, np.ndarray):
            #check for nans or infinites:
            bool_mask_mean_nan_inf = ~np.isfinite(mean_input_array)
            bool_mask_variance_nan_inf = ~np.isfinite(variance_input_array)

            #We want to replace all nans and infinite values
            mean_value, variance_value = PolytopeEvaluator.get_static_estimate_value_only(number_groups, group_index)
            if bool_mask_mean_nan_inf.any():
                mean_input_array[bool_mask_mean_nan_inf] = mean_value

            if bool_mask_variance_nan_inf.any():
                variance_input_array[bool_mask_variance_nan_inf] = variance_value

            return mean_input_array, variance_input_array


    def generate_batch_polytope_package_inputs(self):
        """
        Equality constraints need to be expressed seperately, it will cause errors if you express an equality constraint as two inequality constraints
        :return:
        """

        list_batch_min_val_index = self.list_current_lower_bounds
        list_batch_max_val_index = self.list_current_upper_bounds

        list_A_matrices = []
        list_b_vectors = []
        list_A2_matrices = []
        list_b2_vectors = []

        for batch_index, batch_list_entry in enumerate(self.list_list_full_agg_constraint_tuples):

            np_lhs, np_rhs, list_relation_types = PolytopeEvaluator.convert_masked_agg_tuple_list_into_LE_system(
                list_tuple_agg_constraints=batch_list_entry)
            np_lhs, np_rhs, list_relation_types = PolytopeEvaluator.convert_system_in_all_LE_constraints(np_lhs=np_lhs,
                                                                                                         np_rhs=np_rhs,
                                                                                                         list_relation_types=list_relation_types)
            np_lhs_ineq, np_rhs_ineq, np_lhs_eq, np_rhs_eq = PolytopeEvaluator.seperate_equality_constraints(np_lhs, np_rhs)

            #remove all single variable containts in the inequalities, they should be reflected in the min_val/max_val list
            boolean_single_variable = np.abs(np.sum(np_lhs_ineq, axis=1))>1

            if np_lhs_ineq.size>0:
                list_A_matrices.append(np_lhs_ineq[boolean_single_variable])
                list_b_vectors.append(np_rhs_ineq[boolean_single_variable].flatten())
            else:
                list_A_matrices.append(None)
                list_b_vectors.append(None)

            if np_lhs_eq.size>0:
                list_A2_matrices.append(np_lhs_eq)
                list_b2_vectors.append(np_rhs_eq.flatten())
            else:
                list_A2_matrices.append(None)
                list_b2_vectors.append(None)

        return list_batch_min_val_index, list_batch_max_val_index, list_A_matrices, list_b_vectors, list_A2_matrices, list_b2_vectors

    def generate_batch_mhar_package_inputs(self):
        """
        Equality constraints need to be expressed seperately, it will cause errors if you express an equality constraint as two inequality constraints
        :return:
        """

        list_lhs_ineq_matrices = []
        list_rhs_ineq_vectors = []
        list_lhs_eq_matrices = []
        list_rhs_eq_vectors = []
        list_chebyshev_center = []

        for batch_index, batch_list_entry in enumerate(self.list_list_full_agg_constraint_tuples):

            np_lhs, np_rhs, list_relation_types = PolytopeEvaluator.convert_masked_agg_tuple_list_into_LE_system(
                list_tuple_agg_constraints=batch_list_entry)
            np_lhs, np_rhs, list_relation_types = PolytopeEvaluator.convert_system_in_all_LE_constraints(np_lhs=np_lhs,
                                                                                                         np_rhs=np_rhs,
                                                                                                         list_relation_types=list_relation_types)

            np_lhs_ineq, np_rhs_ineq, np_lhs_eq, np_rhs_eq = PolytopeEvaluator.seperate_equality_constraints(np_lhs, np_rhs)

            #mhar can not deal with a [1, 1, 1, 1] >= 0 if there is also a [1, 1, 1, 1]==1 condition, so we have to filter these out
            bool_mask_invalid_condition = np.sum(np.abs(np_lhs_ineq), axis=1)>=self.number_groups
            np_lhs_ineq = np_lhs_ineq[~bool_mask_invalid_condition]
            np_rhs_ineq = np_rhs_ineq[~bool_mask_invalid_condition]
            #print(bool_mask)

            if np_lhs_eq.size==0:
                np_lhs_eq = None
            if np_rhs_eq.size==0:
                np_rhs_eq = None

            list_lhs_ineq_matrices.append(np_lhs_ineq)
            list_rhs_ineq_vectors.append(np_rhs_ineq)

            list_lhs_eq_matrices.append(np_lhs_eq)
            list_rhs_eq_vectors.append(np_rhs_eq)

            #calculate cheb center
            np_chebyshev_center = PolytopeEvaluator.chebyshev_center(A_in=np_lhs_ineq, b_in=np_rhs_ineq, A_eq = np_lhs_eq, b_eq = np_rhs_eq)
            #print(np_chebyshev_center)
            list_chebyshev_center.append(np_chebyshev_center)

        return list_lhs_ineq_matrices, list_rhs_ineq_vectors, list_lhs_eq_matrices, list_rhs_eq_vectors, list_chebyshev_center

    @staticmethod
    def generate_number_encoding(float_input):
        decimal = 4
        if abs(float_input)<=0.0001:
            return "0.0"
        else:
            return f'{round(float_input,decimal):.4f}'

    @staticmethod
    def seperate_equality_constraints(np_lhs, np_rhs):
        #filter out potential redundant duplicates

        np_initial_merge = np.concatenate([np_lhs, np_rhs], axis=1)
        np_initial_merge_unique = np.unique(np_initial_merge, axis=0)

        np_lhs_cleaned = np_initial_merge_unique[:, :-1]
        np_rhs_cleaned = np_initial_merge_unique[:, [-1]]

        #identify potential duplicates
        np_merged_cleaned = np.concatenate([np_lhs_cleaned, np_rhs_cleaned], axis=1)

        set_encountered_fingerprints = set()
        dict_row_meta_unique = {}
        dict_row_meta_duplicate = {}

        for idx_count, row in enumerate(np_merged_cleaned):
            #row_finger_print = ''.join([f'{Decimal(entry):.2f}' for entry in row.tolist()])
            row_finger_print = ''.join([PolytopeEvaluator.generate_number_encoding(entry) for entry in row.tolist()])
            #row_finger_print_negative = ''.join([f'{str(entry*(-1)) if abs(entry)<=0.001 else str(0)}' for entry in row.tolist()])
            row_finger_print_negative = ''.join(
                [PolytopeEvaluator.generate_number_encoding(entry*(-1)) for entry in row.tolist()])

            #order is important
            list_fingerprint = [row_finger_print, row_finger_print_negative]
            list_fingerprint.sort()
            sorted_finger_print=''.join(list_fingerprint)

            if sorted_finger_print in set_encountered_fingerprints:
                if sorted_finger_print in dict_row_meta_unique:
                    tmp_dict = dict_row_meta_unique.pop(sorted_finger_print) #initially we destroy it
                    dict_row_meta_duplicate[sorted_finger_print] = tmp_dict

                dict_row_meta_duplicate.get(sorted_finger_print)['count']+=1
                dict_row_meta_duplicate.get(sorted_finger_print)['index'].append(idx_count)
            else:
                dict_row_meta_unique[sorted_finger_print] = {
                    'count':1,
                    'index': [idx_count]
                }
            set_encountered_fingerprints.add(sorted_finger_print)

        list_unique_indices = [entry.get("index")[0] for entry in dict_row_meta_unique.values()]
        list_non_unique_indices = [entry.get("index")[0] for entry in dict_row_meta_duplicate.values()]

        np_merged_ineq = np.take(np_merged_cleaned, list_unique_indices, axis=0)
        np_merged_eq = np.take(np_merged_cleaned, list_non_unique_indices, axis=0)

        np_lhs_ineq = np_merged_ineq[:, :-1]
        np_rhs_ineq = np_merged_ineq[:, [-1]]

        np_lhs_eq = np_merged_eq[:, :-1]
        np_rhs_eq = np_merged_eq[:, [-1]]

        return np_lhs_ineq, np_rhs_ineq, np_lhs_eq, np_rhs_eq

    @staticmethod
    def chebyshev_center(A_in, b_in, A_eq=None, b_eq=None, lb=None, ub=None,
                         np_type=np.float64):
        # taken from https://github.com/uumami/mhar/blob/5c777a6687aef27796127060fb74d76d4816a8a1/mhar/chebyshev.py
        # The equality restrictions have zero norm
        A_in_norm = np.matrix(np.sum(A_in ** 2., axis=-1) ** (1. / 2.), dtype=np_type)
        # Create new restriction matrices
        A_in_norm = np.concatenate((A_in, A_in_norm.transpose()), axis=1)

        # Equality restrictions exist
        try:
            eq = A_eq.shape[0]
        except:
            eq = 0
        if eq:
            # The equality restrictions have zero norm. Is transposed to keep order
            A_eq_norm = np.zeros((1, A_eq.shape[0]), dtype=np_type)
            # Create new restriction matrices
            A_eq_norm = np.concatenate((A_eq, A_eq_norm.transpose()), axis=1)
        else:
            A_eq_norm = None

        # Create c
        c = np.concatenate((np.zeros(A_in.shape[1], dtype=np_type), [-1.]))

        r = linprog(c=c,
                    A_ub=A_in_norm,
                    b_ub=b_in,
                    A_eq=A_eq_norm,
                    b_eq=b_eq,
                    bounds=(lb, ub),
                    method='revised simplex')

        status = {0: 'Optimization proceeding nominally.',
                  1: 'Iteration limit reached.',
                  2: 'Problem appears to be infeasible.',
                  3: ' Problem appears to be unbounded.',
                  4: ' Numerical difficulties encountered.',
                  }[r.status]
        print('\nSimplex Status for the Chebyshev Center\n', status)

        return np.array(r.x[:-1], ndmin=2).transpose()


    def estimate_uniform_marginal_normalized_population_parameter_set_hit_and_run_torch(self, column_index):

        number_samples = self.sample_size_mhar_package

        list_batch_min_val_index = self.list_current_lower_bounds
        list_batch_max_val_index = self.list_current_upper_bounds

        list_lhs_ineq_matrices, list_rhs_ineq_vectors, list_lhs_eq_matrices, list_rhs_eq_vectors, list_chebyshev_center \
            = self.generate_batch_mhar_package_inputs()

        list_normalized_marginal_means_polytope = []  # list of means for each polytope (each row in np_already_sampled represents a (part) of a polytope
        list_normalized_marginal_variances_polytope = []  # list of variances for each polytope
        for batch_index in range(len(self.list_list_full_agg_constraint_tuples)):
            torch_lhs_ineq_matrix = torch.from_numpy(list_lhs_ineq_matrices[batch_index]).float().to(
            self.availabe_device)
            torch_rhs_ineq_matrix = torch.from_numpy(list_rhs_ineq_vectors[batch_index]).float().to(
            self.availabe_device)
            torch_lhs_eq_matrix = torch.from_numpy(list_lhs_eq_matrices[batch_index]).float().to(
            self.availabe_device)
            torch_rhs_eq_matrix = torch.from_numpy(list_rhs_eq_vectors[batch_index]).float().to(
            self.availabe_device)
            torch_chebyshev_center = torch.from_numpy(list_chebyshev_center[batch_index]).float().to(
            self.availabe_device)

            #print(torch_lhs_ineq_matrix)
            #print(torch_rhs_ineq_matrix)
            #print(torch_lhs_eq_matrix)
            #print(torch_rhs_eq_matrix)
            #print(torch_chebyshev_center)

            torch_samples = walk(z=number_samples,  # Padding Parameter
                     ai=torch_lhs_ineq_matrix,  # Inequality matrix
                     bi=torch_rhs_ineq_matrix,  # Inequaliy restrictions
                     ae=torch_lhs_eq_matrix,  # Equality Matrix
                     be=torch_rhs_eq_matrix,  # Equality restriction
                     x_0=torch_chebyshev_center,  # Inner Point of the polytope
                     T=1,  # Number of iid Iterations
                     device=str(self.availabe_device),#self.availabe_device,  # Device to use cpu or cuda
                     warm=0,  # Number of iid iterations to burn before saving samples
                     seed=None,  # Automatic random seed
                     thinning=None,  # Automatic thinning factor of O(n^3)
                     check=False
                     )

            #print(torch_samples)
            #print("------------")

            samples_non_violation = torch_samples.cpu().detach().numpy()
            np_relevant_variable_column = samples_non_violation[:, column_index]

            np_relevant_variable_column_normalized = PolytopeEvaluator.normalize_relevant_column_for_samples(
                np_relevant_variable_column,
                list_batch_min_val_index,
                list_batch_max_val_index,
                current_batch_index=batch_index,
                relevant_column_index=column_index)

            np_mean_val, np_var_val = PolytopeEvaluator.calculate_normalized_population_parameter(
                np_relevant_variable_column_normalized=np_relevant_variable_column_normalized)

            list_normalized_marginal_means_polytope.append(np_mean_val)
            list_normalized_marginal_variances_polytope.append(np_var_val)

        return np.array(list_normalized_marginal_means_polytope), np.array(list_normalized_marginal_variances_polytope)


    def sample_complete_polytope_deterministic(self):
        """
        Based on git://github.com/DavidWalz/polytope-sampling.git
        :param number_samples:
        :return:
        """

        list_batch_min_val_index, list_batch_max_val_index, list_A_matrices, list_b_vectors, list_A2_matrices, list_b2_vectors \
            = self.generate_batch_polytope_package_inputs()

        batch_index=0
        A1 = list_A_matrices[batch_index]
        b1 = list_b_vectors[batch_index]
        print(A1)
        print(b1)
        print("###########")
        samples_deterministic = polytope.chebyshev_center(A1,b1)

        return samples_deterministic


    def sample_complete_polytope_uniformly(self, number_samples):
        """
        Based on git://github.com/DavidWalz/polytope-sampling.git

        This samples from the initial polytope WITHOUT any depedent regression steps
        """

        list_batch_min_val_index, list_batch_max_val_index, list_A_matrices, list_b_vectors, list_A2_matrices, list_b2_vectors \
            = self.generate_batch_polytope_package_inputs()

        list_normalized_marginal_means_polytope = []  # list of means for each polytope (each row in np_already_sampled represents a (part) of a polytope
        list_normalized_marginal_variances_polytope = []  # list of variances for each polytope
        batch_index=0
        lower = list_batch_min_val_index[batch_index]
        upper = list_batch_max_val_index[batch_index]
        A1 = list_A_matrices[batch_index]
        b1 = list_b_vectors[batch_index]
        A2 = list_A2_matrices[batch_index]
        b2 = list_b2_vectors[batch_index]
        samples_non_violation = polytope.sample(n_points=number_samples, lower=lower, upper=upper, A1=A1, b1=b1,
                                                    A2=A2, b2=b2)
        return samples_non_violation


    def estimate_uniform_marginal_normalized_population_parameter_set_hit_and_run(self, column_index):

        number_samples = self.sample_size_polytope_package

        list_batch_min_val_index, list_batch_max_val_index, list_A_matrices, list_b_vectors, list_A2_matrices, list_b2_vectors\
            = self.generate_batch_polytope_package_inputs()

        list_normalized_marginal_means_polytope = []  # list of means for each polytope (each row in np_already_sampled represents a (part) of a polytope
        list_normalized_marginal_variances_polytope = []  # list of variances for each polytope
        for batch_index in range(len(self.list_list_full_agg_constraint_tuples)):
            lower = list_batch_min_val_index[batch_index]
            upper = list_batch_max_val_index[batch_index]
            A1 = list_A_matrices[batch_index]
            b1 = list_b_vectors[batch_index]
            A2 = list_A2_matrices[batch_index]
            b2 = list_b2_vectors[batch_index]
            try:
                samples_non_violation = polytope.sample(n_points=number_samples, lower=lower, upper=upper, A1=A1, b1=b1, A2=A2, b2=b2)
                np_relevant_variable_column = samples_non_violation[:, column_index]

                np_relevant_variable_column_normalized = PolytopeEvaluator.normalize_relevant_column_for_samples(np_relevant_variable_column,
                                                                        list_batch_min_val_index,
                                                                        list_batch_max_val_index,
                                                                        current_batch_index=batch_index,
                                                                        relevant_column_index=column_index)

                np_mean_val, np_var_val = PolytopeEvaluator.calculate_normalized_population_parameter(
                    np_relevant_variable_column_normalized=np_relevant_variable_column_normalized)
            except:
                if column_index==0:
                    raise ValueError(f'Input Error, the provided initial polytope is not solvable {self.list_list_full_agg_constraint_tuples}')
                #if the above throws an error it means we found a solution and we just return plcaeholder values
                np_mean_val = 0.9999
                np_var_val = 0.000005
            #print("--------")
            #print(type(np_mean_val))
            #print(np_mean_val.shape)

            list_normalized_marginal_means_polytope.append(np_mean_val)
            list_normalized_marginal_variances_polytope.append(np_var_val)

        #print(list_normalized_marginal_means_polytope)
        #print(list_normalized_marginal_variances_polytope)

        return np.array(list_normalized_marginal_means_polytope), np.array(list_normalized_marginal_variances_polytope)

    @staticmethod
    def normalize_relevant_column_for_samples(np_relevant_column_sample, list_batch_min_val_index, list_batch_max_val_index, current_batch_index, relevant_column_index):

        normalizing_scaling_factor = np.array(list_batch_max_val_index[current_batch_index][relevant_column_index]) - np.array(
            list_batch_min_val_index[current_batch_index][relevant_column_index])
        normalizing_intercept_term = np.array(list_batch_min_val_index[current_batch_index][relevant_column_index])

        #to avoid too large numbers
        if normalizing_scaling_factor < 0.0005:
            normalizing_scaling_factor = 0

        np_relevant_variable_column_normalized = (np_relevant_column_sample - np.ones_like(
            np_relevant_column_sample) * normalizing_intercept_term) / normalizing_scaling_factor

        return np_relevant_variable_column_normalized

    @staticmethod
    def calculate_normalized_population_parameter(np_relevant_variable_column_normalized):
        return np.mean(np_relevant_variable_column_normalized), np.var(np_relevant_variable_column_normalized)

    def estimate_uniform_marginal_normalized_population_parameter_set_rejection_sampling(self, column_index, debug_non_normalized=False):
        """
        This returns the parameters for NORMALIZED samples, i.e. we can sample from [0,1], only afterwards the
        real boundaries given by the polytope are incorporated using a linear transformation X_trans=a*x+b
        The actor however generates actions [0,1] (Which are then transformed in the environment)
        :param column_index:
        :param list_min_val_index:
        :param list_max_val_index:
        :return:
        """
        np_sampled = np.array(self.list_list_all_samples)
        number_samples = 100_000

        np_already_sampled = np_sampled[:,:column_index].astype(float)
        number_free_variables = self.number_groups-column_index

        list_batch_min_val_index = self.list_current_lower_bounds
        list_batch_max_val_index = self.list_current_upper_bounds

        list_normalized_marginal_means_polytope = [] # list of means for each polytope (each row in np_already_sampled represents a (part) of a polytope
        list_normalized_marginal_variances_polytope = [] # list of variances for each polytope
        for batch_index, row in enumerate(np_already_sampled):
            #the last loop will always have zero variance, since everything is determined by then
            scaling_factor = np.sum(row)

            tmp_duplicates = np.tile(row, (number_samples,1))
            samples = PolytopeEvaluator.sample_scaled_dirichlet(scaling_factor=(1-scaling_factor), number_variables=number_free_variables,
                                                                       number_samples=number_samples, list_mask=[1] * number_free_variables)
            tmp_merged_samples = np.concatenate((tmp_duplicates, samples), axis=1)

            #t0 = time.time()
            np_penalty_matrix = PolytopeEvaluator.generate_penalty_matrix(tmp_merged_samples, self.list_list_full_agg_constraint_tuples[batch_index])

            #violation mask
            zero_matrix = np.zeros_like(np_penalty_matrix)
            tmp_bool_matrix = ~np.isclose(zero_matrix, np_penalty_matrix)
            any_violation_in_allocation = np.any(tmp_bool_matrix, axis=1)

            #only take samples which are not in violation
            samples_non_violation = tmp_merged_samples[~any_violation_in_allocation]

            np_relevant_variable_column = samples_non_violation[:, column_index]

            np_relevant_variable_column_normalized = PolytopeEvaluator.normalize_relevant_column_for_samples(
                np_relevant_variable_column,
                list_batch_min_val_index,
                list_batch_max_val_index,
                current_batch_index=batch_index,
                relevant_column_index=column_index)

            np_mean_val, np_var_val = PolytopeEvaluator.calculate_normalized_population_parameter(
                np_relevant_variable_column_normalized=np_relevant_variable_column_normalized)
            list_normalized_marginal_means_polytope.append(np_mean_val)
            list_normalized_marginal_variances_polytope.append(np_var_val)

        #print(list_normalized_marginal_means_polytope)
        #print(list_normalized_marginal_variances_polytope)
        return np.array(list_normalized_marginal_means_polytope), np.array(list_normalized_marginal_variances_polytope)
