#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import time
import threading as thread

import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../')
sys.path.append('hexapod_robot')
sys.path.append('hexapod_explorer')

# import hexapod robot and explorer
import HexapodRobot
import HexapodExplorer

# import communication messages
from messages import *


def p(num, places=1):
    return f'{num:.2f}'

class Explorer:
    """ Class to represent an exploration agent
    """

    def __init__(self, robotID=0):

        """ VARIABLES
        """
        # occupancy grid map of the robot ... possibly extended initialization needed in case of 'm1' assignment
        gridmap = OccupancyGrid()
        gridmap.resolution = 0.1
        gridmap.width = 100
        gridmap.height = 100
        gridmap.origin = Pose(Vector3(-5.0, -5.0, 0.0), Quaternion(1, 0, 0, 0))
        gridmap.data = 0.5 * np.ones((gridmap.height, gridmap.width))
        self.gridmap = gridmap
        self.robot_size = 0.5
        # current frontiers
        self.frontiers = None
        # current path
        self.path = Path()
        # trajectory
        self.trajectory = None
        self.path_to_go = None
        # stopping condition
        self.stop = False
        # timing delays
        self.timing = {"mapping": 1,
                       "planning": 20,
                       "trajectory_following": 1,
                       "graph": 5}

        """Connecting the simulator
        """
        # instantiate the robot
        self.robot = HexapodRobot.HexapodRobot(robotID)
        # ...and the explorer used in task t1c-t1e
        self.explor = HexapodExplorer.HexapodExplorer()

    def start(self):
        """ method to connect to the simulated robot and start the navigation, localization, mapping and planning
        """
        # turn on the robot
        self.robot.turn_on()

        # start navigation thread
        self.robot.start_navigation()

        # start the mapping thread
        try:
            mapping_thread = thread.Thread(target=self.mapping)
            mapping_thread.start()
        except:
            print("Error: unable to start mapping thread")
            sys.exit(1)

        # start planning thread
        try:
            planning_thread = thread.Thread(target=self.planning)
            planning_thread.start()
        except:
            print("Error: unable to start planning thread")
            sys.exit(1)
        # start trajectory following
        try:
            traj_follow_thread = thread.Thread(target=self.trajectory_following)
            traj_follow_thread.start()
        except:
            print("Error: unable to start planning thread")
            sys.exit(1)

    def __del__(self):
        # turn off the robot
        self.robot.stop_navigation()
        self.robot.turn_off()

    def mapping(self):
        """ Mapping thread for fusing the laser scans into the grid map
        """
        while not self.stop:
            time.sleep(self.timing["mapping"])
            # fuse the laser scan
            laser_scan = self.robot.laser_scan_
            odometry = self.robot.odometry_
            self.gridmap = self.explor.fuse_laser_scan(self.gridmap, laser_scan, odometry)

    def planning(self):
        """ Planning thread that takes the constructed gridmap, find frontiers, and select the next goal with the navigation path
        """
        while not self.stop:
            print("Finding new plan.")
            # obstacle growing
            gridmap_processed = self.explor.grow_obstacles(self.gridmap, self.robot_size)
            # frontier calculation
            # self.frontiers = self.explor.find_free_edge_frontiers(self.gridmap, self.robot.laser_scan_, multiple=False)
            self.frontiers = self.explor.find_free_edge_frontiers(self.gridmap, self.robot.laser_scan_, multiple=True)

            # sorted_frontiers = self.closest_frontier(gridmap_processed)
            sorted_frontiers = self.richest_frontier(gridmap_processed)
            print("mutual\n\n")

            # plan path
            self.plan_path(gridmap_processed, sorted_frontiers)

            # new plan found
            if self.trajectory:
                if len(self.trajectory.poses) > 0 or self.robot.navigation_goal:
                    time.sleep(self.timing["planning"])
            # try to find new plan
            else:
                time.sleep(1)

    def trajectory_following(self):
        """trajectory following thread that assigns new goals to the robot navigation thread
        """
        while not self.stop:
            # reached goal and need a new one
            if not self.robot.navigation_goal:
                # move to next goal
                prev_navigation_goal = self.next_navigation_goal()
                if prev_navigation_goal:
                    print("Reached navigation_goal: ",
                          p(prev_navigation_goal.position.x), p(prev_navigation_goal.position.y))
            odometry = self.robot.odometry_
            if odometry:
                self.path.poses.append(odometry.pose)
            time.sleep(self.timing["trajectory_following"])

    def next_navigation_goal(self):
        prev_navigation_goal = self.robot.navigation_goal
        if self.trajectory:
            if len(self.trajectory.poses) > 0:
                self.robot.navigation_goal = self.trajectory.poses.pop(0)
        return prev_navigation_goal

    def circle_I(self, center, I_matrix):
        R = int(self.robot.laser_scan_.range_max)
        cx = round(center.x / self.gridmap.resolution) + self.gridmap.width // 2
        cy = round(center.y / self.gridmap.resolution) + self.gridmap.height // 2
        gain = 0
        for x in range(-R, R):
            Y = int((R**2 - (x)**2)**0.5)
            for y in range(-Y, Y+1):
                point = x+cx, y+cy
                if 0 <= point[0] < self.gridmap.width and 0 <= point[1] < self.gridmap.height:
                    gain += I_matrix[point]
        return gain

    def richest_frontier(self, gridmap_processed):
        if not self.robot.odometry_ or not self.frontiers or not gridmap_processed:
            return
        p_function = np.vectorize(lambda p: p * np.log(p))
        one_p_function = np.vectorize(lambda p:  (1-p) * np.log((1-p)))
        I_matrix = p_function(self.gridmap.data) + one_p_function(self.gridmap.data)
        I_matrix = -1 * I_matrix
        path_goal_metric = []
        for goal in self.frontiers:
            I = self.circle_I(goal.position, I_matrix)
            path_goal_metric.append((None, goal, I))
        sorted_frontiers = sorted(path_goal_metric, key=lambda frontier: frontier[2])
        goal_idx = 0
        while goal_idx < len(sorted_frontiers):
            path, goal, metric = sorted_frontiers[goal_idx]
            success, path_to_go, dist = self.explor.plan_path(gridmap_processed, self.robot.odometry_.pose, goal,
                                                              self.robot_size)
            if success:
                sorted_frontiers[goal_idx] = (path_to_go, goal, dist)
                print("Mutual plan found, goal:", p(goal.position.x), p(goal.position.y), dist)
                break
            goal_idx += 1
        return sorted_frontiers

    def closest_frontier(self, gridmap_processed):
        if not self.robot.odometry_ or not self.frontiers:
            return
        start = self.robot.odometry_.pose
        path_goal_dist = []
        frontier_idx = 0
        for goal in self.frontiers:
            success, path_to_go, dist = self.explor.plan_path(gridmap_processed, start, goal, self.robot_size)
            if success:
                path_goal_dist.append((path_to_go, goal, dist))
            frontier_idx += 1
        sorted_frontiers = sorted(path_goal_dist, key=lambda frontier: frontier[2])
        path_to_go, goal, dist = sorted_frontiers[0]
        print("Closest plan found, goal:", p(goal.position.x), p(goal.position.y), dist)
        return sorted_frontiers

    def plan_path(self, gridmap_processed, sorted_frontiers):
        if not sorted_frontiers:
            return
        path_to_go, goal, metric = sorted_frontiers[0]
        idx = 0
        while not path_to_go and idx < len(sorted_frontiers):
            path_to_go, goal, metric = sorted_frontiers[idx]
            idx += 1
        print("Plan found, goal:", p(goal.position.x), p(goal.position.y), metric)
        # all navigation goals without simplification
        self.path_to_go = path_to_go
        # Simplified plans, minimum navigation goals
        self.trajectory = self.explor.simplify_path(gridmap_processed, path_to_go)
        # Plan starts with current location, pop it
        self.trajectory.poses.pop(0)
        start = self.robot.odometry_.pose
        print("Trajectory:", p(start.position.x), p(start.position.y), "\n",
              *[(p(a.position.x), p(a.position.y)) for a in self.trajectory.poses])
        self.next_navigation_goal()

if __name__ == "__main__":
    # instantiate the robot
    ex0 = Explorer()
    # start the locomotion
    ex0.start()

    # continuously plot the map, targets and plan (once per second)
    while (1):
        fig, ax = plt.subplots()
        plt.ion()
        plt.cla()
        # plot the gridmap
        if ex0.gridmap.data is not None:
            ex0.gridmap.plot(ax)
        # plot the navigation path
        if ex0.path is not None:
            ex0.path.plot(ax)

        if ex0.frontiers is not None:
            x = [f.position.x for f in ex0.frontiers]
            y = [f.position.y for f in ex0.frontiers]
            ax.plot(x, y, "yo")

        if ex0.path_to_go is not None:
            x = [f.position.x for f in ex0.path_to_go.poses]
            y = [f.position.y for f in ex0.path_to_go.poses]
            ax.plot(x, y, "b.")


        plt.xlabel('x[m]')
        plt.ylabel('y[m]')
        ax.set_aspect('equal', 'box')
        plt.show()
        # to throttle the plotting pause for 1s
        plt.pause(ex0.timing["graph"])
        plt.close()
