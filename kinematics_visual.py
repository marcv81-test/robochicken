import visual
from kinematics import *

class VisualLimb(Limb):
    """Visual robotic arm/leg model"""

    def __init__(self,**kwargs):
        self._initialize_first(**kwargs)
        self._initialize_rods()
        self._initialize_second()

    def forward_kinematics(self, angles):
        Limb.forward_kinematics(self, angles)
        self.draw()

    def inverse_kinematics(self, target_end_point):
        Limb.inverse_kinematics(self, target_end_point)
        self.draw_target()

    def _initialize_rods(self):
        """Initialize the limb"""
        self._rods = [None] * self._sections_count
        for i in range(self._sections_count):
            self._rods[i] = visual.cylinder(radius = 0.05);
        self._ball = visual.sphere(radius = 0.1, color = visual.color.red)

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

    def draw_target(self):
        self._ball.pos = self._target_end_point

class VisualMultipod(Multipod):
    """Visual multipod model"""

    def __init__(self, **kwargs):
        kwargs["leg_class"] = VisualLimb
        self._initialize(**kwargs)
