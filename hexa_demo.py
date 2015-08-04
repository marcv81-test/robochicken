import math
from kinematics import *
from kinematics_visual import *
import joystick
from visual import *

scene.forward = [1, 0, 0]
scene.up = [0, 0, -1]
scene.range = 5

arrow(axis = [1, 0, 0], color = color.red)
arrow(axis = [0, 1, 0], color = color.green)
arrow(axis = [0, 0, 1], color = color.blue)

joystick = joystick.Joystick("/dev/input/js0")
hexapod = VisualMultipod()

#hexapod = VisualMultipod(
#        leg_kwargs = {
#            "lengths": [0.25, 1, 2, 0.5],
#            "axes": [[0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0]],
#            "angles_limits": [
#                [-tau / 8, tau / 8],
#                [-3 * tau / 8, tau / 8],
#                [0 * tau / 8, 4 * tau / 8],
#                [-tau / 8, tau / 16]]},
#        legs_count = 8)

t = 0.0
dt = 0.04
while True:
    rate(25)
    t += dt
    joystick.update()
    x = -0.4 * joystick.axis_states["ry"]
    y = 0.4 * joystick.axis_states["rx"]
    z = 0.2 * (joystick.axis_states["y"] + 1.0)
    hexapod.direct_control(x, y, z)
