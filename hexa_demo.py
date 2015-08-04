import math
from kinematics import *
from kinematics_visual import *
import joystick

joystick = joystick.Joystick("/dev/input/js0")
hexapod = VisualMultipod()

#hexapod = VisualMultipod(
#        leg_kwargs = {
#            "lengths": [0.25, 1, 2, 0.5],
#            "axes": [[0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0]],
#            "angles_limits": [
#                [-tau / 8, tau / 8],
#                [-tau / 8, 3 * tau / 8],
#                [-3 * tau / 8, -tau / 8],
#                [-tau / 16, tau / 8]]},
#        legs_count = 8)

t = 0.0
dt = 0.04
while True:
    visual.rate(25)
    t += dt
    joystick.update()
    x = 0.5 * joystick.axis_states["rx"]
    y = 0.5 * joystick.axis_states["ry"]
    z = 0.3 * (joystick.axis_states["y"] + 1.0)
    hexapod.direct_control(x, y, z)
