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
        self.robot_size = 0.1
        # current frontiers
        self.frontiers = None
        # current path
        self.path = Path()
        # trajectory
        self.trajectory = None
        # stopping condition
        self.stop = False
        # timing delays
        self.timing = {"mapping": 5,
                       "planning": 10,
                       "trajectory_following": 3}


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
            print("mapping continue")
            # fuse the laser scan
            laser_scan = self.robot.laser_scan_
            odometry = self.robot.odometry_
            self.gridmap = self.explor.fuse_laser_scan(self.gridmap, laser_scan, odometry)

    def planning(self):
        """ Planning thread that takes the constructed gridmap, find frontiers, and select the next goal with the navigation path
        """
        sufficient_trajectory_poses_n = 1
        while not self.stop:
            if self.trajectory:
                if len(self.trajectory.poses) >= sufficient_trajectory_poses_n:
                    time.sleep(self.timing["planning"])
                    print("planning continue")
            else:
                print("finding plan")
                time.sleep(5)
            # obstacle growing
            gridmap_processed = self.explor.grow_obstacles(self.gridmap, self.robot_size)
            # frontier calculation
            self.frontiers = self.explor.find_free_edge_frontiers(self.gridmap)
            # goal selection
            start = None
            odometry = self.robot.odometry_
            if odometry:
                start = odometry.pose
            # path planning
            if self.frontiers:
                sorted_frontiers = self.closest_frontier(self.frontiers, gridmap_processed)
                path_planning_limit = 5
                frontier_idx = 1
                while frontier_idx < path_planning_limit and frontier_idx < len(sorted_frontiers):
                    goal, goal_dist = sorted_frontiers[frontier_idx]
                    if self.frontier_occupied(goal, gridmap_processed):
                        path_planning_limit += 1
                        frontier_idx += 1
                        print("Frontier goal not accessible.")
                        continue
                    flag, path_to_go = self.explor.plan_path(gridmap_processed, start, goal, self.robot_size)
                    print(flag, goal_dist)
                    if path_to_go:
                        self.trajectory = self.explor.simplify_path(gridmap_processed, path_to_go)
                        print(*[(int(a.position.x), int(a.position.y)) for a in self.trajectory.poses])
                        break
                    else:
                        frontier_idx += 1

    def trajectory_following(self):
        """trajectory following thread that assigns new goals to the robot navigation thread
        """
        prev_navigation_goal = self.robot.navigation_goal
        while not self.stop:
            time.sleep(self.timing["trajectory_following"])
            if self.close_enough():
                if prev_navigation_goal:
                    print("reached navigation_goal: ", int(prev_navigation_goal.position.x), int(prev_navigation_goal.position.y))
                if self.trajectory:
                    if len(self.trajectory.poses) > 0:
                        goal = self.trajectory.poses.pop(0)
                        prev_navigation_goal = goal
                        # goto_reactive sets robot.navigation_goal
                        self.robot.goto_reactive(goal)
            else:
                pass
                # print(self.robot.odometry_.pose.position.x,
                #       self.robot.odometry_.pose.position.y,
                #       self.robot.navigation_goal.position.x,
                #       self.robot.navigation_goal.position.y)
            odometry = self.robot.odometry_
            if odometry:
                self.path.poses.append(odometry.pose)

    def close_enough(self):
        LIM = 3 * self.robot_size
        if self.robot.odometry_:
            if self.robot.navigation_goal:
                dist = self.robot.odometry_.pose.dist(self.robot.navigation_goal)
                if dist <= LIM:
                    print("CLOSE ENOUGH", dist)
                    return True
                else:
                    return False
            print("NO GOAL")
            return True
        return False


    def closest_frontier(self, frontiers, gridmap_processed):
        if not self.robot.odometry_ or not frontiers:
            return None, None
        position = self.robot.odometry_.pose.position
        f = []
        for a in frontiers:
            goal = a.position
            f.append(np.linalg.norm([goal.x - position.x, goal.y - position.y]))
        sorted_frontiers = sorted([(frontiers[i], d) for i, d in enumerate(f)], key=lambda frontier: frontier[1])
        return sorted_frontiers

    def frontier_occupied(self, goal, gridmap_processed):
        x, y = goal.position.x, goal.position.y
        print("\t\t", x, y)
        x = x / gridmap_processed.resolution + (gridmap_processed.width // 2)
        y = y / gridmap_processed.resolution + (gridmap_processed.height // 2)
        x, y = round(x), round(y)
        print("\t\t", x, y)
        return gridmap_processed.data[y][x]



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
            ax.plot(x, y, "ro")

        if ex0.trajectory is not None:
            x = [f.position.x for f in ex0.trajectory.poses]
            y = [f.position.y for f in ex0.trajectory.poses]
            ax.plot(x, y, "b.")

        plt.xlabel('x[m]')
        plt.ylabel('y[m]')
        ax.set_aspect('equal', 'box')
        plt.show()
        # to throttle the plotting pause for 1s
        plt.pause(5)
        plt.close()
