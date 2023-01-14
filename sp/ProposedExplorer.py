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
                       "graph": 10}


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
            # print("mapping continue")
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
            self.closest_frontier(gridmap_processed)
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

    def closest_frontier(self, gridmap_processed):
        if not self.robot.odometry_ or not self.frontiers:
            return
        start = self.robot.odometry_.pose
        position = start.position
        f = []
        for a in self.frontiers:
            goal = a.position
            f.append(np.linalg.norm([goal.x - position.x, goal.y - position.y]))
        sorted_frontiers = sorted([(self.frontiers[i], d) for i, d in enumerate(f)], key=lambda frontier: frontier[1])
        # goal selection
        print("Frontiers found.")
        frontier_idx = 0
        while frontier_idx < len(sorted_frontiers):
            goal, goal_dist = sorted_frontiers[frontier_idx]
            frontier_idx += 1
            # second argument allows fallback
            if goal_dist <= 3 * self.robot_size and frontier_idx < len(sorted_frontiers):
                print("TOO CLOSE",
                      p(goal_dist), p(goal.position.x), p(goal.position.y), frontier_idx)
                continue
            # path planning
            flag, path_to_go = self.explor.plan_path(gridmap_processed, start, goal, self.robot_size)
            # print(frontier_idx, "Path found:", flag)
            print("Plan found, goal:", p(goal.position.x), p(goal.position.y), goal_dist)
            self.path_to_go = path_to_go
            self.trajectory = self.explor.simplify_path(gridmap_processed, path_to_go)
            self.trajectory.poses.pop(0)
            print("Trajectory:", p(start.position.x), p(start.position.y), "\n",
                  *[(p(a.position.x), p(a.position.y)) for a in self.trajectory.poses])
            self.next_navigation_goal()
            break

    def frontier_occupied(self, goal, gridmap_processed):
        x, y = goal.position.x, goal.position.y
        print("\t\t", x, y)
        x = x / gridmap_processed.resolution + (gridmap_processed.width // 2)
        y = y / gridmap_processed.resolution + (gridmap_processed.height // 2)
        x, y = round(x), round(y)
        print("\t\t", x, y)
        return gridmap_processed.data[y][x]

    def close_enough(self):
        LIM = self.robot_size
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
