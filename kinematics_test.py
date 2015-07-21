import numpy as np
import numpy.testing as npt

from kinematics import *

def assert_rotation_almost_equal(a, b):
    x = [1, 0, 0]
    y = [0, 1, 0]
    z = [0, 0, 1]
    npt.assert_almost_equal(np.dot(a, x), np.dot(b, x))
    npt.assert_almost_equal(np.dot(a, y), np.dot(b, y))
    npt.assert_almost_equal(np.dot(a, z), np.dot(b, z))

def assert_rigid_motion_almost_equal(a, b):
    assert_rotation_almost_equal(a._rotation, b._rotation)
    npt.assert_almost_equal(a._translation, b._translation)

def test_rotation():

    a = rotation([1, 0, 0], 0)
    b = np.identity(3)
    assert_rotation_almost_equal(b, a)

    c = rotation([1, 0, 0], tau / 4)
    d = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    assert_rotation_almost_equal(c, d)

class TestRigidMotion:

    def test_translation_only(self):

        a = RigidMotion(rotation([0, 0, 1], tau / 2), [1, 0, 0])
        b = [-1, 0, 0]
        npt.assert_almost_equal(b, a.translation_only())

        c = RigidMotion(rotation([0, 0, 1], tau / 2), [0, 1, 0])
        d = [0, -1, 0]
        npt.assert_almost_equal(d, c.translation_only())

    def test_compose(self):

        a = RigidMotion(rotation([0, 0, 1], tau / 2), [1, 0, 0])
        b = RigidMotion(rotation([0, 0, 1], tau / 4), [1, 0, 0])

        c = RigidMotion(rotation([0, 0, 1], -tau / 4), [1, 1, 0])
        assert_rigid_motion_almost_equal(c, a.compose(b))

        d = RigidMotion(rotation([0, 0, 1], -tau / 4), [0, 0, 0])
        assert_rigid_motion_almost_equal(d, b.compose(a))

class TestJacobianSolver:

    def test_jacobian_matrix(self):

        matrix = [[1, 0, 3], [0, 2, 2], [1, 2, 1]]
        f = lambda x: np.dot(matrix, x)

        # The Jacobian matrix of a linear map is the matrix of the linear map itself
        f_solver = JacobianSolver(f)
        npt.assert_almost_equal(matrix, f_solver.jacobian_matrix([0, 0, 0]))
        npt.assert_almost_equal(matrix, f_solver.jacobian_matrix([1, 2, 3]))

    def test_converge(self):

        matrix = [[1, 0, 3], [0, 2, 2], [1, 2, 1]]
        f = lambda x: np.dot(matrix, x)

        # For a linear map, the solver converges in a single iteration
        f_solver = JacobianSolver(f)
        npt.assert_almost_equal([1, 1, 0], f_solver.converge([0, 0, 0], [1, 2, 3]))

if __name__ == "__main__":
    npt.run_module_suite()
