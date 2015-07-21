import math
from kinematics import *
from kinematics_visual import *

legs = [None] * 6
for i in range(6):
    initial_rotation = rotation([0, 0, 1], tau / 12 + i * tau / 6)
    legs[i] = VisualLimb(initial_rigid_motion = RigidMotion(initial_rotation, [1, 0, 1]))

legs_baseline_position = [None] * 6
baseline_position = RigidMotion(translation = [1.5, 0, -1.7])
for i in range(6):
    legs_baseline_position[i] = legs[i]._initial_rigid_motion.compose(baseline_position).translation_only()

t = 0.0
dt = 0.01
while True:
    visual.rate(100)
    t += dt
    displacement_x = 0.75 * math.sin(5 * t)
    displacement_z = 0.5 * math.cos(5 * t)
    factor = 1.0
    for i in range(6):
        target_position = list(legs_baseline_position[i])
        if (i % 2 == 0):
            target_position[0] = target_position[0] + displacement_x
            if (displacement_z > 0):
                target_position[2] = target_position[2] + displacement_z
        else:
            target_position[0] = target_position[0] - displacement_x
            if (displacement_z < 0):
                target_position[2] = target_position[2] - displacement_z
        legs[i].inverse_kinematics(target_position)
        legs[i].draw()
