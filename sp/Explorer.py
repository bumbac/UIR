#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

import sys
import os
import copy
import time

import math
import numpy as np
import threading as thread

sys.path.append('messages')
sys.path.append('hexapod_robot')
sys.path.append('hexapod_explorer')

#import hexapod robot and explorer
import HexapodRobot 
import HexapodExplorer

#import communication messages
from messages import *

class Explorer:
    def __init__(self, robotID = 0):
        #instantiate the robot
        self.robot = HexapodRobot.HexapodRobot(robotID)
        self.explor = HexapodExplorer.HexapodExplorer()

    def start(self):
        #turn on the robot 
        self.robot.turn_on()

        #start navigation thread
        self.robot.start_navigation()

    def __del__(self):
        #turn off the robot
        self.robot.stop_navigation()
        self.robot.turn_off()


if __name__ == "__main__":
    #instantiate the exploration agent
    ex0 = Explorer(0)
    #ex1 = Explorer(1)

    #start the agent
    ex0.start()
    #ex1.start()

    #main loop
    while(1):
        time.sleep(1)
