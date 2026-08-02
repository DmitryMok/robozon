"""Контроллер вращающегося вала демо-шибера: крутит roller_motor с заданной
постоянной скоростью (рад/с), передаётся через controllerArgs.
"""
import sys

from controller import Robot

robot = Robot()
velocity = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
motor = robot.getDevice("roller_motor")
motor.setPosition(float("inf"))
motor.setVelocity(velocity)

timestep = int(robot.getBasicTimeStep())
while robot.step(timestep) != -1:
    pass
