#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

#import messages
from messages import *

DELTA_DISTANCE = 0.12
C_TURNING_SPEED = 5
C_AVOID_SPEED = 10

class HexapodController:
    def __init__(self):
        pass


    def goto(self, goal, odometry, collision):
        """Method to steer the robot towards the goal position given its current 
           odometry and collision status
        Args:
            goal: Pose of the robot goal
            odometry: Perceived odometry of the robot
            collision: bool of the robot collision status
        Returns:
            cmd: Twist steering command
        """
        # zero velocity steering command
        cmd_msg = Twist()

        #TODO: if input data are valid, calculate the steering command
        
        return cmd_msg            


    def goto_reactive(self, goal, odometry, collision, laser_scan):
        """Method to steer the robot towards the goal position while avoiding 
           contact with the obstacles given its current odometry, collision 
           status and laser scan data
        Args:
            goal: Pose of the robot goal
            odometry: Perceived odometry of the robot
            collision: bool of the robot collision status
            laser_scan: LaserScan data perceived by the robot
        Returns:
            cmd: Twist steering command
        """
        #zero velocity steering command
        cmd_msg = Twist()

        #TODO: if input data are valid, calculate the steering command
        
        return cmd_msg
