import numpy as np
import numpy.testing as npt

from jacobian import *

class TestJacobianSolver:

    def test_jacobian_matrix(self):

        matrix = [[1, 0, 3], [0, 2, 2], [1, 2, 1]]
        f = lambda x: np.dot(matrix, x)

        # The Jacobian matrix of a linear map is the matrix of the linear map itself
        f_solver = JacobianSolver(function = f)
        npt.assert_almost_equal(matrix, f_solver.jacobian_matrix(input_vector = [0, 0, 0]))
        npt.assert_almost_equal(matrix, f_solver.jacobian_matrix(input_vector = [1, 2, 3]))

class TestJacobianInverseSolver:

    def test_limit_input_fix_vector(self):

        f = lambda x: x
        f_solver = JacobianInverseSolver(function = f, max_input_fix = 1)

        # A norm above the max is altered
        npt.assert_almost_equal([1, 0, 0], f_solver._limit_input_fix_vector([2, 0, 0]))

        # A norm below the max is not altered
        npt.assert_almost_equal([0.5, 0, 0], f_solver._limit_input_fix_vector([0.5, 0, 0]))

    def test_converge(self):

        matrix = [[1, 0, 3], [0, 2, 2], [1, 2, 1]]
        f = lambda x: np.dot(matrix, x)

        # For a linear map, the solver converges in a single iteration
        f_solver = JacobianInverseSolver(function = f, max_input_fix = 1000)
        npt.assert_almost_equal([1, 1, 0], f_solver.converge(
                input_vector = [0, 0, 0], target_output_vector = [1, 2, 3]))

class TestDampedLeastSquaresSolver:

    def test_converge(self):

        matrix = [[1, 0, 3], [0, 2, 2], [1, 2, 1]]
        f = lambda x: np.dot(matrix, x)

        # For a linear map, the solver converges after enough iterations
        f_solver = DampedLeastSquaresSolver(function = f, constant = 1)
        input_vector = [0, 0, 0]
        for n in range(50):
            input_vector = f_solver.converge(
                    input_vector = input_vector,
                    target_output_vector = [1, 2, 3])
        npt.assert_almost_equal([1, 1, 0], input_vector)

if __name__ == "__main__":
    npt.run_module_suite()
