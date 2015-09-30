import math
import argparse
from visual import *
from kinematics import *
from kinematics_visual import *
import joystick

parser = argparse.ArgumentParser(description = "Display a joystick-controlled multipod.")
parser.add_argument("--model", help = "hexapod by default, or 'spider'")
args = parser.parse_args()

scene.range = 5
scene.forward = [1, 0, 0]
scene.up = [0, 0, -1]
arrow(axis = [1, 0, 0], color = color.red)
arrow(axis = [0, 1, 0], color = color.green)
arrow(axis = [0, 0, 1], color = color.blue)

joystick = joystick.Joystick("/dev/input/js0")

if (args.model == "spider"):
    multipod = VisualMultipod(
            leg_kwargs = {
                "lengths": [0.25, 1, 2, 0.5],
                "axes": [[0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0]],
                "angles_limits": [
                    [-tau / 8, tau / 8],
                    [-3 * tau / 8, tau / 8],
                    [0 * tau / 8, 4 * tau / 8],
                    [-tau / 8, tau / 16]]},
            legs_count = 8)
else:
    multipod = VisualMultipod()

t = 0.0
dt = 0.04
while True:
    rate(25)
    t += dt
    joystick.update()
    x = -0.4 * joystick.axis_states["ry"]
    y = 0.4 * joystick.axis_states["rx"]
    z = 0.2 * (joystick.axis_states["y"] + 1.0)
    a = tau / 32.0 * (joystick.axis_states["x"])
    multipod.direct_control(x, y, z, a)
