#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

import sys
import os
import math
import numpy as np

sys.path.append('hexapod_robot')
 
#import hexapod robot 
import HexapodRobot as hexapod

#import communication messages
from messages import *

if __name__=="__main__":

    robot = hexapod.HexapodRobot(0)

    #turn on the robot 
    robot.turn_on()

    #start locomotion
    robot.start_locomotion()

    #prepare figure
    plt.ion()
    fig, ax = plt.subplots()

    #laser scan
    while True:
        #get the current laser scan
        laserscan = robot.laser_scan_

        #check for validity of the data
        if laserscan is not None:
            plt.cla()
            #plot the laser scan           
            laserscan.plot(ax)
            #plot the robot frame
            ax.quiver([0],[0],[1],[0], color='r')
            ax.quiver([0],[0],[0],[1], color='g')

            plt.axis("equal")
            plt.show()
            plt.pause(0.1)

