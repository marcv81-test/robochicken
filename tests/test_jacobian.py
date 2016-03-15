import unittest
import numpy as np
import numpy.testing as npt

from robotics.jacobian import *

class JacobianMatrixTestCase(unittest.TestCase):

    def test_jacobian_matrix(self):

        matrix = ((1, 0, 3), (0, 2, 2), (1, 2, 1))
        f = lambda x: np.dot(matrix, x)
        f_solver = JacobianSolver(function = f)

        # The Jacobian matrix of a linear map at any input vector is the
        # matrix of the linear map.
        npt.assert_almost_equal(matrix, f_solver._jacobian_matrix(
                input_vector = (0, 0, 0)))
        npt.assert_almost_equal(matrix, f_solver._jacobian_matrix(
                input_vector = (1, 2, 3)))

class JacobianInverseSolverTestCase(unittest.TestCase):

    def test_converge(self):

        matrix = ((1, 0, 3), (0, 2, 2), (1, 2, 1))
        f = lambda x: np.dot(matrix, x)
        f_solver = JacobianInverseSolver(function = f)

        # For a linear map the unconstrained solver reaches the target
        # output vector after a single iteration.
        npt.assert_almost_equal((1, 1, 0), f_solver.converge(
                input_vector = (0, 0, 0),
                target_output_vector = (1, 2, 3)))

    def test_converge_max_input_fix(self):

        matrix = ((1, 0, 3), (0, 2, 2), (1, 2, 1))
        f = lambda x: np.dot(matrix, x)
        f_solver = JacobianInverseSolver(
                function = f,
                max_input_fix = 0.5)

        # The input vector has to go from (0, 0, 0) to (1, 1, 0). The solver
        # input fix is limited to 0.5 per component. The solver gets halfway
        # there after a single iteration.
        input_vector = (0, 0, 0)
        input_vector = input_vector = f_solver.converge(
               input_vector = input_vector,
               target_output_vector = (1, 2, 3))
        npt.assert_almost_equal((0.5, 0.5, 0), input_vector)

        # And reaches the goal after a second iteration.
        input_vector = f_solver.converge(
                input_vector = input_vector,
                target_output_vector = (1, 2, 3))
        npt.assert_almost_equal((1, 1, 0), input_vector)

    def test_converge_max_output_error(self):

        matrix = ((1, 0, 3), (0, 2, 2), (1, 2, 1))
        f = lambda x: np.dot(matrix, x)
        f_solver = JacobianInverseSolver(
                function = f,
                max_output_error = np.sqrt(14) / 2)

        # The output vector has to go from (0, 0, 0) to (1, 2, 3). The
        # initial output vector error norm is hence sqrt(14). The solver
        # output error norm is limited to sqrt(14)/2. The solver gets
        # halfway there after a single iteration.
        input_vector = (0, 0, 0)
        input_vector = f_solver.converge(
                input_vector = input_vector,
                target_output_vector = (1, 2, 3))
        npt.assert_almost_equal((0.5, 0.5, 0), input_vector)

        # And reaches the goal after a second iteration.
        input_vector = f_solver.converge(
                input_vector = input_vector,
                target_output_vector = (1, 2, 3))
        npt.assert_almost_equal((1, 1, 0), input_vector)

class DampedLeastSquaresSolverTest(unittest.TestCase):

    def test_converge(self):

        matrix = ((1, 0, 3), (0, 2, 2), (1, 2, 1))
        f = lambda x: np.dot(matrix, x)

        # For a linear map the solver reaches the target output vector
        # after enough iterations.
        f_solver = DampedLeastSquaresSolver(function = f, constant = 1)
        input_vector = (0, 0, 0)
        for n in xrange(50):
            input_vector = f_solver.converge(
                    input_vector = input_vector,
                    target_output_vector = (1, 2, 3))
        npt.assert_almost_equal((1, 1, 0), input_vector)
