import joystick

joystick = joystick.Joystick("/dev/input/js0")

print ("Axes: " + ", ".join(joystick.axis_map))
print ("Buttons: " + ", ".join(joystick.button_map))

while True:
    joystick.read_events()
    print joystick.axis_states["x"], joystick.button_states["a"]
