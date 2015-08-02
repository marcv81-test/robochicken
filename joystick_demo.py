import joystick
from time import sleep

joystick = joystick.Joystick("/dev/input/js0")

print ("Axes: " + ", ".join(joystick.axis_map))
print ("Buttons: " + ", ".join(joystick.button_map))

while True:
    joystick.update()
    print("X axis: %.2f" % joystick.axis_states["x"])
    print("A button: %i" % joystick.button_states["a"])
    sleep(1)
