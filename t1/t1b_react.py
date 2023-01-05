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

    #start navigation thread
    robot.start_navigation()
    time.sleep(3)

    #assign goal for navigation
    goals = [
        Pose(Vector3(3.5,3.5,0),Quaternion(1,0,0,0)),
        Pose(Vector3(0,0,0),Quaternion(1,0,0,0)),
    ]

    path = Path()
    #go from goal to goal
    for goal in goals:
        robot.goto_reactive(goal)
        while robot.navigation_goal is not None:
            #sample the current odometry
            if robot.odometry_ is not None:
                path.poses.append(robot.odometry_.pose)
            #wait
            time.sleep(0.1)

        #check the robot distance to goal
        odom = robot.odometry_.pose
        #compensate for the height of the robot as we are interested only in achieved planar distance 
        odom.position.z = 0 
        #calculate the distance
        dist = goal.dist(odom)
        print("[t1b_eval] distance to goal: %.2f" % dist)

    robot.stop_navigation()
    robot.turn_off()

    #plot the robot path
    fig, ax = plt.subplots()
    path.plot(ax, 30)
    plt.xlabel('x[m]')
    plt.ylabel('y[m]')
    plt.axis('equal')
    plt.show()
