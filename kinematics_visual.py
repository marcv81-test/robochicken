import visual
from kinematics import *

class VisualLimb(Limb):
    """Visual robotic arm/leg model"""

    def __init__(self, *args, **kwargs):
        Limb.__init__(self, *args, **kwargs)
        self._rods = [None] * self._sections_count
        for i in range(self._sections_count):
            self._rods[i] = visual.cylinder(radius = 0.05);

    def draw(self):
        """Draw the limb"""
        points = [None] * (self._sections_count + 1)
        rigid_motion = self._initial_rigid_motion
        points[0] = rigid_motion.translation_only()
        for i in range(self._sections_count):
            rigid_motion = rigid_motion.compose(RigidMotion(
                rotation(self._axes[i], self._angles[i]),
                [self._lengths[i], 0, 0]))
            points[i + 1] = rigid_motion.translation_only()
            self._rods[i].pos = points[i]
            self._rods[i].axis = points[i + 1] - points[i]
