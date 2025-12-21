import RPi.GPIO as GPIO

TOUCH_PIN = 17
GPIO.setup(TOUCH_PIN, GPIO.IN)

def is_touched():
    return GPIO.input(TOUCH_PIN) == 1
