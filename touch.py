# touch.py
import time

from gpiozero import Button

touch = Button(17, pull_up=False)  # BCM pin


def is_touched():
    return touch.is_pressed


if __name__ == '__main__':
    while True:
        print(is_touched())
        time.sleep(2)
