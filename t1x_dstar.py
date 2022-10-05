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


def plot_path(ax, path, clr):
    """ simple method to draw the path
    """
    if path is not None:
        poses = np.asarray([(pose.position.x,pose.position.y) for pose in path.poses])
        ax.plot(poses[:,0], poses[:,1], '.',color=clr)
        ax.plot(poses[:,0], poses[:,1], '-',color=clr)

def plot_dstar_map(ax, grid_map, rhs_=None, g_=None):
    """method to plot the gridmap with rhs and g values
    """
    #plot the gridmap
    gridmap.plot(ax)
    plt.grid(True, which='major')

    rhs = rhs_.reshape(grid_map.height, grid_map.width)
    g = g_.reshape(grid_map.height, grid_map.width)
    #annotate the gridmap graph with g and rhs values
    if g is not None and rhs is not None:
        for i in range(0, gridmap.width):
            for j in range(0, gridmap.height):
                ax.annotate("%.2f" % g[i,j], xy=(i+0.1, j+0.2), color="red")
                ax.annotate("%.2f" % rhs[i,j], xy=(i+0.1, j+0.6), color="blue")

def check_collision(gridmap, pose, scenario):
    """ method to check whether the robot is in collision with newly discovered obstacle
    """
    if to_grid(gridmap, pose) in scenario:
        return True
    else:
        return False

def to_grid(grid_map, pose):
    """method to transform world coordinates to grid coordinates
    """
    cell = ((np.asarray((pose.position.x, pose.position.y)) - (grid_map.origin.position.x, grid_map.origin.position.y)) / grid_map.resolution)
    return  (int(cell[0]), int(cell[1]))


if __name__=="__main__":

    #define planning problems:
    scenarios = [([(1, 0), (1, 1), (1, 2), (1, 3), (2, 3), (3, 3), (4, 3), (4, 2), (4, 1), (5, 1), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (5, 6), (4, 6), (3, 6), (2, 6), (1, 6), (7, 1), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7), (8, 8), (7, 8), (6, 8), (5, 8), (4, 8), (3, 8), (2, 8), (1, 8)]),
                 ([(1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (3, 9), (3, 8), (3, 7), (3, 6), (3, 5), (3, 4), (3, 3), (3, 2), (3, 1), (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (7, 9), (7, 8), (7, 7), (7, 6), (7, 5), (7, 4), (7, 3), (7, 2), (7, 1), (8, 1)]),
                 ([(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (7, 1), (8, 1), (6, 1), (8, 2), (8, 3), (8, 4), (7, 4), (7, 5), (7, 6), (9, 6), (8, 6), (8, 7), (8, 8), (7, 8), (6, 8), (5, 8), (4, 8), (3, 8), (2, 8), (1, 8), (1, 7), (1, 6), (1, 4), (1, 3), (1, 2)])]

    start = Pose(Vector3(4.5, 4.5, 0),Quaternion(0,0,0,1))
    goal = Pose(Vector3(9.5, 5.5, 0),Quaternion(0,0,0,1))

    #prepare plot
    fig, ax = plt.subplots(figsize=(10,10))
    plt.ion()

    cnt = 0
    #fetch individual scenarios
    for scenario in scenarios:
        robot_odometry = Odometry()
        robot_odometry.pose = start

        robot_path = Path()
        robot_path.poses.append(start)

        #instantiate the explorer robot
        explor = explorer.HexapodExplorer()

        #instantiate the map
        gridmap = OccupancyGrid()
        gridmap.resolution = 1
        gridmap.width = 10
        gridmap.height = 10
        gridmap.origin = Pose(Vector3(0,0,0), Quaternion(0,0,0,1))
        gridmap.data = np.zeros((gridmap.height*gridmap.width))
        
        while not (robot_odometry.pose.position.x == goal.position.x and
               robot_odometry.pose.position.y == goal.position.y):
            #plan the route from start to goal
            path, rhs, g = explor.plan_path_incremental(gridmap, robot_odometry.pose, goal)
            if path == None:
                print("Destination unreachable")
                break
            if len(path.poses) < 1:
                print("There is nowhere to go")
                break

            #check for possibility of the move
            if check_collision(gridmap, path.poses[1], scenario):
                #add the obstacle into the gridmap 
                cell = to_grid(gridmap, path.poses[1])
                data = gridmap.data.reshape(gridmap.height, gridmap.width)
                data[cell[1],cell[0]] = 1
                gridmap.data = data.flatten()
            else:
                #move the robot
                robot_odometry.pose = path.poses[1]

            robot_path.poses.append(robot_odometry.pose)

            #plot it
            plt.cla()
            plot_dstar_map(ax, gridmap, rhs, g)
            plot_path(ax, path, 'r')
            plot_path(ax, robot_path, 'b')
            robot_odometry.pose.plot(ax)
            plt.xlabel('x[m]')
            plt.ylabel('y[m]')
            plt.axis('square')

            #optional save of the image for debugging purposes
            #fig.savefig("result/%04d.png" % cnt, bbox_inches='tight')
            cnt += 1
            plt.show()
            plt.pause(0.1)

