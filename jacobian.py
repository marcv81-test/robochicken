import numpy as np

class JacobianSolver:
    """Base class for numeric solvers using Jacobian matrices"""

    def __init__(self, function, input_delta = 0.001):
        """Constructor"""
        self._function = function
        self._input_delta = input_delta

    def jacobian_transpose_matrix(self, input_vector, output_vector = None):
        """Jacobian transpose matrix of the function at the input vector"""
        if (output_vector == None):
            output_vector = self._function(input_vector)
        input_size = len(input_vector)
        output_size = len(output_vector)
        matrix = np.zeros((input_size, output_size))
        for i in range(input_size):
            altered_input_vector = list(input_vector)
            altered_input_vector[i] = altered_input_vector[i] + self._input_delta
            altered_output_vector = self._function(altered_input_vector)
            output_delta_vector = altered_output_vector - output_vector
            matrix[i] = output_delta_vector / self._input_delta
        return matrix

    def jacobian_matrix(self, **kwargs):
        """Jacobian matrix of the function at the input vector"""
        return np.transpose(self.jacobian_transpose_matrix(**kwargs))

class JacobianInverseSolver(JacobianSolver):
    """Numeric solver using the Jacobian inverse technique"""

    def __init__(self, max_input_fix, **kwargs):
        """Constructor"""
        self._max_input_fix = max_input_fix
        JacobianSolver.__init__(self, **kwargs)

    def jacobian_inverse_matrix(self, **kwargs):
        """Jacobian inverse matrix of the function at the input vector"""
        return np.linalg.pinv(self.jacobian_matrix(**kwargs))

    def converge(self, input_vector, target_output_vector, output_vector = None):
        """
        Attempts to calculate an improved input vector
        so that the output vector converges toward the target.
        """
        if (output_vector == None):
            output_vector = self._function(input_vector)
        output_error_vector = target_output_vector - output_vector
        matrix = self.jacobian_inverse_matrix(
                input_vector = input_vector,
                output_vector = output_vector)
        input_fix_vector = np.dot(matrix, output_error_vector)
        input_fix_vector = self._limit_input_fix_vector(input_fix_vector)
        return input_vector + input_fix_vector

    def _limit_input_fix_vector(self, input_fix_vector):
        """Limits the input fix vector"""
        global_ratio = 1
        for i in range(len(input_fix_vector)):
            ratio = abs(input_fix_vector[i] / self._max_input_fix)
            if ratio > global_ratio:
                global_ratio = ratio
        if global_ratio > 1:
            input_fix_vector = np.asarray(input_fix_vector) / global_ratio
        return input_fix_vector
