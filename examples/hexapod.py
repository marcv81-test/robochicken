import os.path

from robotics.joystick import *
from robotics.kinematics.leg import *

scene.range = 5
scene.forward = [1, 0, 0]
scene.up = [0, 0, -1]
arrow(axis = [1, 0, 0], color = color.red)
arrow(axis = [0, 1, 0], color = color.green)
arrow(axis = [0, 0, 1], color = color.blue)

lookup_table = LookupTable(
        input_specifications = [
            {'from': -2, 'to': 2, 'points': 9},
            {'from': -2, 'to': 2, 'points': 9},
            {'from': -2, 'to': 2, 'points': 9}],
        output_size = 3)

if os.path.isfile('lookup_leg.npy'):
    lookup_table.load('lookup_leg.npy')
else:
    LookupTableLeg.populate(lookup_table)
    lookup_table.save('lookup_leg')

legs_count = 6

joystick = Joystick("/dev/input/js1")

legs = []
leg_translation = Displacement(translation = (1, 0, 0))
for i in xrange(legs_count):
    angle = i * tau / legs_count
    leg_rotation = Displacement(rotation = Rotation.axis_angle((0, 0, 1), angle))
    leg_displacement = leg_rotation.compose(leg_translation)
    leg = LookupTableLeg(
            initial_displacement = leg_displacement,
            lookup_table = lookup_table)
    leg.initialize_draw()
    legs.append(leg)

t = 0.0
dt = 0.04
while True:
    rate(25)
    t += dt
    joystick.update()
    x = -2 * joystick.axis_states["ry"]
    y = 2 * joystick.axis_states["rx"]
    z = 2 * (joystick.axis_states["y"])
    for i in xrange(legs_count):
        legs[i].endpoint_inverse_kinematics((x, y, z))
        legs[i].draw()
