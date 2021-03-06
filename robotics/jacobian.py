import numpy as np

class JacobianSolver:
    """Base class for numeric solvers using Jacobian matrices"""

    def __init__(self, function,
            max_input_fix = None,
            max_output_error = None,
            input_delta = 0.001):
        """Constructor"""
        self._function = function
        self._max_input_fix = max_input_fix
        if max_input_fix == None: self._max_input_fix = None
        else: self._max_input_fix = float(max_input_fix)
        if max_output_error == None: self._max_output_error = None
        else: self._max_output_error = float(max_output_error)
        self._input_delta = float(input_delta)

    def _jacobian_transpose_matrix(self, input_vector, output_vector = None):
        """Jacobian transpose matrix of the function at the input vector"""
        input_vector = np.asfarray(input_vector)
        if (output_vector == None):
            output_vector = self._function(input_vector)
        output_vector = np.asfarray(output_vector)
        matrix = np.zeros((len(input_vector), len(output_vector)))
        for i in xrange(len(input_vector)):
            altered_input_vector = input_vector.copy()
            altered_input_vector[i] += self._input_delta
            altered_output_vector = self._function(altered_input_vector)
            altered_output_vector = np.asfarray(altered_output_vector)
            output_delta_vector = altered_output_vector - output_vector
            matrix[i] = output_delta_vector / self._input_delta
        return matrix

    def _jacobian_matrix(self, **kwargs):
        """Jacobian matrix of the function at the input vector"""
        return np.transpose(self._jacobian_transpose_matrix(**kwargs))

    def converge(self, input_vector, target_output_vector, output_vector = None):
        """Attempt to calculate an improved input vector so that the
        output vector converges toward the target.
        """
        input_vector = np.asfarray(input_vector)
        target_output_vector = np.asfarray(target_output_vector)
        if (output_vector == None):
            output_vector = self._function(input_vector)
        output_vector = np.asfarray(output_vector)
        matrix = self._solver_matrix(
                input_vector = input_vector,
                output_vector = output_vector)
        output_error_vector = target_output_vector - output_vector
        JacobianSolver._limit_norm(output_error_vector, self._max_output_error)
        input_fix_vector = np.dot(matrix, output_error_vector)
        JacobianSolver._limit_component(input_fix_vector, self._max_input_fix)
        return input_vector + input_fix_vector

    @staticmethod
    def _limit_component(vector, max_component):
        """Scale a vector so that no components exceeds a maximum"""
        if max_component == None: return
        assert vector.dtype == np.float_
        highest_component = np.amax(np.absolute(vector))
        if highest_component > max_component:
            np.multiply(vector, max_component / highest_component, vector)

    @staticmethod
    def _limit_norm(vector, max_norm):
        """Scale a vector so that the norm does not exceed a maximum"""
        if max_norm == None: return
        assert vector.dtype == np.float_
        norm = np.linalg.norm(vector)
        if norm > max_norm:
            np.multiply(vector, max_norm / norm, vector)

class JacobianInverseSolver(JacobianSolver):
    """Numeric solver using the Jacobian inverse technique"""

    def __init__(self, **kwargs):
        """Constructor"""
        JacobianSolver.__init__(self, **kwargs)

    def _solver_matrix(self, **kwargs):
        """Jacobian inverse matrix of the function at the input vector"""
        return np.linalg.pinv(self._jacobian_matrix(**kwargs))

class DampedLeastSquaresSolver(JacobianSolver):
    """Numeric solver using the Damped Least Squares (DLS) technique"""

    def __init__(self, constant, **kwargs):
        """Constructor"""
        self._constant = constant
        JacobianSolver.__init__(self, **kwargs)

    def _solver_matrix(self, **kwargs):
        """Damped Least Squares matrix"""
        jacobian_transpose_matrix = self._jacobian_transpose_matrix(**kwargs)
        jacobian_matrix = np.transpose(jacobian_transpose_matrix)
        square_matrix = np.dot(jacobian_matrix, jacobian_transpose_matrix)
        size = square_matrix.shape[0]
        identity = np.identity(size)
        return np.dot(
                jacobian_transpose_matrix,
                np.linalg.inv(square_matrix + self._constant**2 * identity))
