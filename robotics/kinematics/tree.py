import numpy as np
import collections
from visual import *

from robotics.displacement import *

class Tree:
    """Kinematic Tree. Can be used as a kinematic chain.

    Each node has a key, a part, a parent, and a list of children.
    The part, parent, and list of children are stored by key in
    dictionaries. This allows arbitrary node access by key.
    """

    def __init__(self, root_displacement):
        """Constructor"""
        self._parts = { 'root': _Root(root_displacement) }
        self._parents = dict()
        self._children = { 'root': list() }

    def add_node(self, key, part, parent = 'root'):
        """Add a node to the tree"""
        if key in self._parts:
            raise ValueError('Key "' + key + '" aready exists')
        self._parts[key] = part
        self._parents[key] = parent
        self._children[key] = list()
        self._children[parent].append(key)

    def prepare_parameters(self):
        """Prepare empty parameters for evaluate"""
        parameters = dict()
        for key in self._parts:
            parameters[key] = dict()
        return parameters

    def evaluate(self, parameters):
        """Walk the tree and evaluate the displacement at each node
        using forward kinematics.
        """
        displacements = dict()
        todo = collections.deque()
        todo.append('root')
        while todo:
            key = todo.popleft()
            displacement = self._parts[key].displacement(**parameters[key])
            if key != 'root':
                previous_displacement = displacements[self._parents[key]]
                displacement = previous_displacement.compose(displacement)
            displacements[key] = displacement
            for child_key in self._children[key]:
                todo.append(child_key)
        return displacements

    def initialize_draw(self):
        """Initialize the visual parts"""
        for key in self._parts:
            if hasattr(self._parts[key], 'initialize_draw'):
                self._parts[key].initialize_draw()

    def uninitialize_draw(self):
        """Delete the visual parts"""
        for key in self._parts:
            if hasattr(self._parts[key], 'initialize_draw'):
                self._parts[key].uninitialize_draw()

    def draw(self, parameters):
        """Draw the visual parts"""
        displacements = self.evaluate(parameters)
        for key in self._parts:
            if hasattr(self._parts[key], 'draw'):
                displacement_before = displacements[self._parents[key]]
                displacement_after = displacements[key]
                self._parts[key].draw(displacement_before, displacement_after)

class RigidLink:
    """Rigid link part"""

    def __init__(self, length):
        self._displacement = Displacement(translation = (length, 0, 0))

    def displacement(self):
        return self._displacement

    def initialize_draw(self):
        self._rod = cylinder(radius = 0.05)

    def uninitialize_draw(self):
        self._rod.visible = False
        del self._rod

    def draw(self, displacement_before, displacement_after):
        point_before = displacement_before.translation
        point_after = displacement_after.translation
        self._rod.pos = np.copy(point_before)
        self._rod.axis = point_after - point_before

class RevoluteJoint:
    """Revolute joint part"""

    def __init__(self, axis, mount_angle):
        self._axis = axis
        self._mount_angle = mount_angle

    def displacement(self, angle):
        angle = angle + self._mount_angle
        return Displacement(rotation = Rotation.axis_angle(self._axis, angle))

class _Root:
    """Part used as the root node of any tree.

    Stores the intial displacement. Shall *not* be used as a regular
    part anywhere else in a tree.
    """

    def __init__(self, displacement):
        self._displacement = displacement

    def displacement(self):
        return self._displacement
