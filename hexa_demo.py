import math
from kinematics import *
from kinematics_visual import *
import joystick

joystick = joystick.Joystick("/dev/input/js0")

legs = [None] * 6
for i in range(6):
    initial_rotation = rotation([0, 0, 1], tau / 12 + i * tau / 6)
    legs[i] = VisualLimb(initial_rigid_motion = RigidMotion(initial_rotation, [1, 0, 1]))

legs_baseline_position = [None] * 6
baseline_position = RigidMotion(translation = [2.2, 0, -1.2])
for i in range(6):
    legs_baseline_position[i] = legs[i]._initial_rigid_motion.compose(baseline_position).translation_only()

t = 0.0
dt = 0.04
while True:
    visual.rate(25)
    joystick.update()
    t += dt
    displacement_x = 0.5 * joystick.axis_states["rx"]
    displacement_y = 0.5 * joystick.axis_states["ry"]
    displacement_z = 0.8 * joystick.axis_states["y"]
    factor = 1.0
    for i in range(6):
        target_position = list(legs_baseline_position[i])
        if (i % 2 == 0):
            target_position[0] = target_position[0] + displacement_x
            target_position[1] = target_position[1] + displacement_y
            if (displacement_z > 0):
                target_position[2] = target_position[2] + displacement_z
        else:
            target_position[0] = target_position[0] - displacement_x
            target_position[1] = target_position[1] - displacement_y
            if (displacement_z < 0):
                target_position[2] = target_position[2] - displacement_z
        legs[i].inverse_kinematics(target_position)
        legs[i].draw()
