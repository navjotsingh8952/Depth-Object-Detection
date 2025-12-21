from gpiozero import DistanceSensor


class Ultrasonic:
    def __init__(self, trig, echo):
        self.sensor = DistanceSensor(
            trigger=trig,
            echo=echo,
            max_distance=2.0
        )

    def distance_cm(self):
        return round(self.sensor.distance * 100, 1)


if __name__ == '__main__':
    ultra_left = Ultrasonic(trig=23, echo=24)

    while True:
        left = ultra_left.distance_cm()
        print(left)
