# touch.py
from gpiozero import Button

touch = Button(17)  # BCM pin


def is_touched():
    return touch.is_pressed
