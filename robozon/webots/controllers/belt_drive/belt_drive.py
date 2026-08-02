"""Контроллер конвейера SimpleConveyor: крутит ленту с заданной скоростью.

Скорость (м/с) передаётся через controllerArgs из PROTO.
"""
import sys

from controller import Robot

robot = Robot()
speed = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
motor = robot.getDevice("belt_motor")
motor.setPosition(float("inf"))
motor.setVelocity(speed)

timestep = int(robot.getBasicTimeStep())
while robot.step(timestep) != -1:
    pass
