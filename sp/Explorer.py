#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import copy
import sys
import time
import threading as thread
import argparse


import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../')
sys.path.append('hexapod_robot')
sys.path.append('hexapod_explorer')

# import hexapod robot and explorer
import HexapodRobot
import HexapodExplorer

from HexapodExplorer import position_to_coordinates, coordinates_to_position

# import communication messages
from messages import *

CLOSE_RANGE = 0.5

def p(num, places=1):
    return f'{num:.2f}'


LASER_RANGE_LIMITATION = 5


class Explorer:
    """ Class to represent an exploration agent
    """

    def __init__(self, robot_size, r, width, height, origin, robotID=0):

        """ VARIABLES
        """
        # occupancy grid map of the robot ... possibly extended initialization needed in case of 'm1' assignment
        # assignment M1
        gridmap = OccupancyGrid()
        gridmap.resolution = r
        gridmap.width = width
        gridmap.height = height
        gridmap.origin = Pose(Vector3(origin[0], origin[1], 0.0), Quaternion(1, 0, 0, 0))
        gridmap.data = 0.5 * np.ones((gridmap.height, gridmap.width))
        self.gridmap = gridmap
        self.robot_size = robot_size
        # current frontiers
        self.frontiers = None
        # current path
        self.path = Path()
        # trajectory

        self.planning_ctr = 0
        self.trajectory = None
        self.path_to_go = None
        # stopping condition
        self.stop = False
        # timing delays
        self.timing = {"mapping": 1,
                       "planning": 5,
                       "trajectory_following": 1,
                       "graph": 1}

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

    def end_mission(self):
        self.stop = True
        self.__del__()

    def planning(self):
        """ Planning thread that takes the constructed gridmap, find frontiers, and select the next goal with the navigation path
        """
        # Planning periodically and rewriting previous trajectory
        while not self.stop:
            print("Finding new plan.")
            # obstacle growing
            gridmap_processed = self.explor.grow_obstacles(self.gridmap, self.robot_size)
            # frontier calculation
            self.frontiers = self.explor.find_free_edge_frontiers(self.gridmap, self.robot.laser_scan_, multiple=True)
            if not self.frontiers:
                self.planning_ctr += 1
            else:
                if len(self.frontiers) < 5:
                    self.planning_ctr += 1
                else:
                    self.planning_ctr = 0

            if self.planning_ctr > 3:
                self.end_mission()
            # assignment P1
            # sorted_frontiers = self.closest_frontier(gridmap_processed)
            # assignment P2 and F3 using mutual information in circle
            sorted_frontiers = self.richest_frontier(gridmap_processed)

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
                self.next_navigation_goal()
            # save current position for printing
            odometry = self.robot.odometry_
            if odometry:
                self.path.poses.append(odometry.pose)
            time.sleep(self.timing["trajectory_following"])

    def next_navigation_goal(self):
        # assign following goal from available goals in trajectory
        if self.trajectory:
            if len(self.trajectory.poses) > 0:
                self.robot.goto_reactive(self.trajectory.poses.pop(0))
        else:
            # extension, wake-up planning thread to find new plan
            pass

    def circle_blocking_information_gain(self, center, I_matrix):
        # limit the range of the laser_scan to give more focus on individual frontiers
        R = int(self.robot.laser_scan_.range_max) // LASER_RANGE_LIMITATION
        start = position_to_coordinates(center, self.gridmap.origin.position, self.gridmap.resolution)
        mutual_information_gain = 0
        # number of samples in the circle
        N = 100
        # calculate the points on the circumreference of a circle in 0,0 origin
        circle = [(np.cos(2 * np.pi / N * X) * R, np.sin(2 * np.pi / N * X) * R) for X in range(N)]
        for x, y in circle:
            shifted_center = copy.deepcopy(center)
            shifted_center.x += x
            shifted_center.y += y
            goal = position_to_coordinates(shifted_center, self.gridmap.origin.position, self.gridmap.resolution)
            if 0 <= goal[0] < self.gridmap.width and 0 <= goal[1] < self.gridmap.height:
                free_line = self.explor.bresenham_line(start, goal)
                for point in free_line:
                    if self.gridmap.data[point[1]][point[0]] > 0.5:
                        break
                    mutual_information_gain += I_matrix[point]
        return mutual_information_gain

    def richest_frontier(self, gridmap_processed):
        # calculates potential informational gain by visiting specific frontier and plans a path to the richest
        # assignment F3 and P2
        if not self.robot.odometry_ or not self.frontiers or not gridmap_processed:
            return
        # first term of the entropy
        p_function = np.vectorize(lambda p: p * np.log(p))
        # second term of the entropy
        one_p_function = np.vectorize(lambda p:  (1-p) * np.log((1-p)))
        # join them - I use vectorization to speedup the process
        # I_matrix has mutual information for each cell
        I_matrix = -1 * (p_function(self.gridmap.data) + one_p_function(self.gridmap.data))
        path_goal_metric = []
        for goal in self.frontiers:
            I = self.circle_blocking_information_gain(goal.position, I_matrix)
            # path not yet found
            path_goal_metric.append((None, goal, I))
        # sort descending on Information gain, higher I is better
        sorted_frontiers = sorted(path_goal_metric, key=lambda frontier: frontier[2], reverse=True)
        return sorted_frontiers

    def closest_frontier(self, gridmap_processed):
        # assignment P1
        if not self.robot.odometry_ or not self.frontiers:
            return
        start = self.robot.odometry_.pose
        path_goal_dist = []
        frontier_idx = 0
        for goal in self.frontiers:
            # find path
            success, path_to_go, dist = self.explor.plan_path(gridmap_processed, start, goal, self.robot_size)
            if success:
                path_goal_dist.append((path_to_go, goal, dist))
            frontier_idx += 1
        # sort ascending on distance
        sorted_frontiers = sorted(path_goal_dist, key=lambda frontier: frontier[2])
        path_to_go, goal, dist = sorted_frontiers[0]
        print("Closest ", end="")
        return sorted_frontiers

    def plan_path(self, gridmap_processed, sorted_frontiers):
        if not sorted_frontiers:
            return
        start = self.robot.odometry_.pose
        # tentatively select the first path
        frontier_idx = 0
        path_to_go = None 
        # select next frontier if path does not exist
        dist = 0
        while not path_to_go and frontier_idx < len(sorted_frontiers):
            _, goal, metric = sorted_frontiers[frontier_idx]
            success, path_to_go, dist = self.explor.plan_path(gridmap_processed, start, goal, self.robot_size)
            # path successfully calculated
            if dist > CLOSE_RANGE:
                # Simplified plans, minimum navigation goals
                self.trajectory, dist = self.explor.simplify_path(gridmap_processed, path_to_go)
                if dist > CLOSE_RANGE:
                    # All navigation goals without simplification
                    self.path_to_go = path_to_go
                    print("Plan found, goal:", p(goal.position.x), p(goal.position.y), dist)
                    # Plan starts with current location, pop it
                    # self.trajectory.poses.pop(0)
                    print("Trajectory:", p(start.position.x), p(start.position.y), "\n",
                        *[(p(a.position.x), p(a.position.y)) for a in self.trajectory.poses])
                    self.next_navigation_goal()
                    return
            # choose following frontier
            frontier_idx += 1
        print("fallback")
        self.turn_around(start)

    def turn_around(self, start):
        # calculate goal in opposite heading of the robot
        yaw, pitch, roll = start.orientation.to_Euler()
        opposite_yaw = (yaw + np.pi) % (2*np.pi)
        goal_x = start.position.x + (2*CLOSE_RANGE * np.cos(opposite_yaw))
        goal_y = start.position.y + (2*CLOSE_RANGE * np.sin(opposite_yaw))
        path = Path()
        goal = Pose(position=Vector3(goal_x, goal_y, 0))
        path.poses.append(goal)
        self.path_to_go = path
        self.trajectory = path
        self.next_navigation_goal()
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-rs', default='0.5', type=float, help='Robot size.')
    parser.add_argument('-r', default='0.1', type=float, help='Minimal resolution.')
    parser.add_argument('-width', default='100', type=int, help='Width of the map in cells.')
    parser.add_argument('-height', default='100', type=int, help='Height of the map in cells.')
    parser.add_argument('-origin', default=[-5, -5], nargs ="+", type=float, help='Origin of the map.')

    args = parser.parse_args()

    print("Robot size:", args.rs)
    print("Resolution:", args.r)
    print("Width:", args.width)
    print("Height:", args.height)
    print("Origin:", args.origin)
    


    # instantiate the robot
    ex0 = Explorer(args.rs, args.r, args.width, args.height, args.origin)
    # start the locomotion
    ex0.start()

    # continuously plot the map, targets and plan (once per second)
    fig, ax = plt.subplots()
    plt.ion()
    while (1):
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

        if ex0.trajectory is not None:
            x = [f.position.x for f in ex0.trajectory.poses]
            y = [f.position.y for f in ex0.trajectory.poses]
            ax.plot(x, y, "m*")


        plt.xlabel('x[m]')
        plt.ylabel('y[m]')
        ax.set_aspect('equal', 'box')
        plt.show()
        # to throttle the plotting pause for 1s
        plt.pause(ex0.timing["graph"])
        # plt.close()
