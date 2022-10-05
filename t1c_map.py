#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

import sys
import os
import math
import numpy as np

sys.path.append('hexapod_robot')
sys.path.append('hexapod_explorer')
 
#import hexapod robot and explorer
import HexapodRobot as hexapod
import HexapodExplorer as explorer

#import communication messages
from messages import *

#switch to select between the simple variant or the hard variant
SIMPLE_VARIANT = True

if __name__=="__main__":

    robot = hexapod.HexapodRobot(0)
    explor = explorer.HexapodExplorer()

    #turn on the robot 
    robot.turn_on()

    #start navigation thread
    robot.start_navigation()

    #assign goal for navigation
    goals = [
        Pose(Vector3(3.5,3.5,0),Quaternion(1,0,0,0)),
        Pose(Vector3(0.0,0.0,0),Quaternion(1,0,0,0)),
    ]

    #prepare the online plot
    plt.ion()
    fig, ax = plt.subplots()

    #prepare the gridmap
    gridmap = OccupancyGrid()
    gridmap.resolution = 0.1

    if SIMPLE_VARIANT:
        gridmap.width = 100
        gridmap.height = 100
        gridmap.origin = Pose(Vector3(-5.0,-5.0,0.0), Quaternion(1,0,0,0))
        gridmap.data = 0.5*np.ones((gridmap.height*gridmap.width))

    #go from goal to goal
    for goal in goals:
        robot.goto_reactive(goal)
        while robot.navigation_goal is not None:
            plt.cla()

            #get the current laser scan and odometry and fuse them to the map
            gridmap = explor.fuse_laser_scan(gridmap, robot.laser_scan_, robot.odometry_)

            #plot the map
            gridmap.plot(ax)
            plt.xlabel('x[m]')
            plt.ylabel('y[m]')
            plt.axis('square')
            plt.show()
            plt.pause(0.1)

    robot.stop_navigation()
    robot.turn_off()
