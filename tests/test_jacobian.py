import unittest
import numpy as np
import numpy.testing as npt

from robotics.jacobian import *

# Shortcut to create arrays of floats
def np_af(x):
    return np.array(x, dtype = np.float_)

class LimitTestCase(unittest.TestCase):

    # No change if the highest component is at the limit
    vector = np_af((1, -2))
    JacobianSolver._limit_component(vector, 2)
    npt.assert_almost_equal((1, -2), vector)

    # Scaling if the highest component exceeds the limit
    vector = np_af((1, -2))
    JacobianSolver._limit_component(vector, 1)
    npt.assert_almost_equal((0.5, -1), vector)

    # No change if the norm is at the limit
    vector = np_af((-6, 8))
    JacobianSolver._limit_norm(vector, 10)
    npt.assert_almost_equal((-6, 8), vector)

    # Scaling if the norm exceeds the limit
    vector = np_af((-6, 8))
    JacobianSolver._limit_norm(vector, 5)
    npt.assert_almost_equal((-3, 4), vector)

class JacobianMatrixTestCase(unittest.TestCase):

    def test_jacobian_matrix(self):

        matrix = np_af(((1, 0, 3), (0, 2, 2), (1, 2, 1)))
        f = lambda x: np.dot(matrix, x)

        # The Jacobian matrix of a linear map is the matrix of the linear map itself
        f_solver = JacobianSolver(function = f)
        npt.assert_almost_equal(matrix, f_solver.jacobian_matrix(
                input_vector = np_af((0, 0, 0))))
        npt.assert_almost_equal(matrix, f_solver.jacobian_matrix(
                input_vector = np_af((1, 2, 3))))

class JacobianInverseSolverTestCase(unittest.TestCase):

    def test_converge(self):

        matrix = np_af(((1, 0, 3), (0, 2, 2), (1, 2, 1)))
        f = lambda x: np.dot(matrix, x)

        # For a linear map, the solver converges in a single iteration
        f_solver = JacobianInverseSolver(function = f)
        npt.assert_almost_equal((1, 1, 0), f_solver.converge(
                input_vector = np_af((0, 0, 0)),
                target_output_vector = np_af((1, 2, 3))))

class DampedLeastSquaresSolverTest(unittest.TestCase):

    def test_converge(self):

        matrix = np_af(((1, 0, 3), (0, 2, 2), (1, 2, 1)))
        f = lambda x: np.dot(matrix, x)

        # For a linear map, the solver converges after enough iterations
        f_solver = DampedLeastSquaresSolver(function = f, constant = 1)
        input_vector = np_af((0, 0, 0))
        for n in range(50):
            input_vector = f_solver.converge(
                    input_vector = input_vector,
                    target_output_vector = np_af((1, 2, 3)))
        npt.assert_almost_equal((1, 1, 0), input_vector)
