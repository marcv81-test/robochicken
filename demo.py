import joystick

from hexapod import *

scene.range = 5
scene.forward = [1, 0, 0]
scene.up = [0, 0, -1]
arrow(axis = [1, 0, 0], color = color.red)
arrow(axis = [0, 1, 0], color = color.green)
arrow(axis = [0, 0, 1], color = color.blue)

joystick = joystick.Joystick("/dev/input/js1")
hexapod = Hexapod(limited_joints = True)
hexapod.initialize_draw()

t = 0.0
dt = 0.04
while True:
    rate(25)
    t += dt
    joystick.update()
    x = -1.5 * joystick.axis_states["ry"]
    y = 1.5 * joystick.axis_states["rx"]
    z = 1.8 * (joystick.axis_states["y"])
    angle = tau / 8 * (joystick.axis_states["x"])
    hexapod.direct_control(x, y, z, angle)
    hexapod.draw()
