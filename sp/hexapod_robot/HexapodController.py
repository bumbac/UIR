#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math

import numpy as np
from messages import *

DELTA_DISTANCE = 0.2
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
        if (goal is not None) and (odometry is not None) and (collision is not None):
            if collision:
                return None

            diff = goal.position - odometry.pose.position
            dist_to_goal = diff.norm()
            # print(dist_to_goal)

            if dist_to_goal < DELTA_DISTANCE:
                return None
            goal_h = np.arctan2(diff.y, diff.x)
            # print(goal_h)
            robot_h = odometry.pose.orientation.to_Euler()[0]
            # print(robot_h)
            diff_h = goal_h - robot_h
            # magic
            diff_h = (diff_h + math.pi) % (2*math.pi) - math.pi

            cmd_msg.linear.x = dist_to_goal
            cmd_msg.angular.z = C_TURNING_SPEED*diff_h

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
        cmd_msg = Twist()
        if (goal is not None) and (odometry is not None) and (collision is not None) and (laser_scan is not None):
            cmd_msg = self.goto(goal=goal, odometry=odometry, collision=collision)
            if cmd_msg is None:
                return cmd_msg
            diff = goal.position - odometry.pose.position
            linear_speed = diff.norm()
            if linear_speed <= DELTA_DISTANCE:
                return None
            goal_h = np.arctan2(diff.y, diff.x)
            robot_h = odometry.pose.orientation.to_Euler()[0]
            dphi = goal_h - robot_h
            dphi = (dphi + math.pi) % (2 * math.pi) - math.pi


            l = len(laser_scan.distances)
            left_points = laser_scan.distances[:l//2]
            right_points = laser_scan.distances[l//2:]

            # scan_left = min(left_points)
            # scan_right = min(right_points)
            # repulsive_force = 0
            # if scan_right > 0 and scan_left > 0:
            #     repulsive_force = 1/scan_left - 1/scan_right

            scan_left = [point for i, point in enumerate(left_points) if laser_scan.range_min < point < laser_scan.range_max]
            scan_right = [point for i, point in enumerate(right_points) if laser_scan.range_min < point < laser_scan.range_max]
            repulsive_force = 1/min(scan_left) - 1/min(scan_right)

            angular_speed_navigation_component = dphi*C_TURNING_SPEED
            angular_speed_avoidance_component = repulsive_force*C_AVOID_SPEED
            angular_speed = angular_speed_avoidance_component + angular_speed_navigation_component

            # cmd_msg.linear.x = linear_speed
            cmd_msg.angular.z = C_TURNING_SPEED * angular_speed

        return cmd_msg
