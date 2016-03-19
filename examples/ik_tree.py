from visual import *

from robotics.joystick import *
from robotics.kinematics.tree import *
from robotics.jacobian import *

class ExampleTree:

    def __init__(self):
        """Constructor"""
        self._initialize_tree()
        self._joints_angles = [0] * 6
        self._endpoint = self.endpoint(self._joints_angles)
        self._default_endpoint = self._endpoint
        self._target_endpoint = self._endpoint
        self._solver = DampedLeastSquaresSolver(
                function = lambda x: self.endpoint(x),
                constant = 0.8,
                max_input_fix = tau / 16,
                max_output_error = 1)

    def endpoint(self, joints_angles):
        """Forward kinematics equation of the tree endpoint"""
        parameters = self._prepare_parameters(joints_angles)
        displacements = self._tree.evaluate(parameters)
        return np.concatenate([
                displacements['d1'].translation,
                displacements['d2'].translation])

    def endpoint_inverse_kinematics(self, target_endpoint):
        """Update the joints angles according to an IK approximation
        to attempt to reach a set endpoint position.
        """
        self._target_endpoint = target_endpoint
        self._joints_angles = self._solver.converge(
                input_vector = self._joints_angles,
                target_output_vector = self._target_endpoint,
                output_vector = self._endpoint)
        self._endpoint = self.endpoint(self._joints_angles)

    def initialize_draw(self):
        """Initialize the visual elements"""
        self._tree.initialize_draw()
        self._ball1 = visual.sphere(radius = 0.1, color = visual.color.red)
        self._ball2 = visual.sphere(radius = 0.1, color = visual.color.blue)

    def uninitialize_draw(self):
        """Delete the visual elements"""
        self._tree.uninitialize_draw()
        self._ball1.visible = False
        self._ball2.visible = False
        del self._ball1
        del self._ball2

    def draw(self):
        """Render the visual elements"""
        parameters = self._prepare_parameters(self._joints_angles)
        self._tree.draw(parameters)
        self._ball1.pos = self._target_endpoint[:3]
        self._ball2.pos = self._target_endpoint[3:]

    def _initialize_tree(self):
        """Initialize the kinematic tree"""
        self._tree = Tree(Displacement())
        self._tree.add_node(
                key = 'root_a_joint',
                part = RevoluteJoint(
                    axis = [0, 0, 1],
                    mount_angle = 0))
        self._tree.add_node(
                key = 'a',
                part = RigidLink(1),
                parent = 'root_a_joint')
        self._tree.add_node(
                key = 'a_b_joint',
                part = RevoluteJoint(
                    axis = [0, 0, 1],
                    mount_angle = 0),
                parent = 'a')
        self._tree.add_node(
                key = 'b',
                part = RigidLink(1),
                parent = 'a_b_joint')
        # First branch
        self._tree.add_node(
                key = 'b_c1_joint',
                part = RevoluteJoint(
                    axis = [0, 0, 1],
                    mount_angle = 0),
                parent = 'b')
        self._tree.add_node(
                key = 'c1',
                part = RigidLink(1),
                parent = 'b_c1_joint')
        self._tree.add_node(
                key = 'c1_d1_joint',
                part = RevoluteJoint(
                    axis = [0, 0, 1],
                    mount_angle = 0),
                parent = 'c1')
        self._tree.add_node(
                key = 'd1',
                part = RigidLink(1),
                parent = 'c1_d1_joint')
        # Second branch
        self._tree.add_node(
                key = 'b_c2_joint',
                part = RevoluteJoint(
                    axis = [0, 0, 1],
                    mount_angle = 0),
                parent = 'b')
        self._tree.add_node(
                key = 'c2',
                part = RigidLink(1),
                parent = 'b_c2_joint')
        self._tree.add_node(
                key = 'c2_d2_joint',
                part = RevoluteJoint(
                    axis = [0, 0, 1],
                    mount_angle = 0),
                parent = 'c2')
        self._tree.add_node(
                key = 'd2',
                part = RigidLink(1),
                parent = 'c2_d2_joint')

    def _prepare_parameters(self, joints_angles):
        """Prepare tree parameters from joints angles"""
        parameters = self._tree.prepare_parameters()
        parameters['root_a_joint']['angle'] = joints_angles[0]
        parameters['a_b_joint']['angle'] = joints_angles[1]
        parameters['b_c1_joint']['angle'] = joints_angles[2]
        parameters['c1_d1_joint']['angle'] = joints_angles[3]
        parameters['b_c2_joint']['angle'] = joints_angles[4]
        parameters['c2_d2_joint']['angle'] = joints_angles[5]
        return parameters

scene.range = 5
scene.forward = [0, 0, 1]
scene.up = [0, 1, 0]

joystick = Joystick("/dev/input/js1")

tree = ExampleTree()
tree.initialize_draw()

t = 0.0
dt = 0.04
while True:
    rate(25)
    t += dt
    joystick.update()
    x1 = -4 *joystick.axis_states["x"]
    y1 = -4 * joystick.axis_states["y"]
    x2 = -4 *joystick.axis_states["rx"]
    y2 = -4 * joystick.axis_states["ry"]
    tree.endpoint_inverse_kinematics([x1, y1, 0, x2, y2, 0])
    tree.draw()
