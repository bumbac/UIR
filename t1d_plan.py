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

#pickle
import pickle

ROBOT_SIZE = 0.3


if __name__=="__main__":

    #instantiate the explorer robot
    explor = explorer.HexapodExplorer()

    #load scenarios
    scenarios =  pickle.load( open( "resources/scenarios.bag", "rb" ) )

    #evaluate individual scenarios
    for gridmap, start, goal in scenarios:
        #obstacle growing
        gridmap_processed = explor.grow_obstacles(gridmap, ROBOT_SIZE)

        #path planning
        path = explor.plan_path(gridmap_processed, start, goal)

        #path simplification
        path_simple = explor.simplify_path(gridmap_processed, path)

        #plot the result
        def plot_path(path, ax, clr):
            """ simple method to draw the path
            """
            if path is not None:
                poses = np.asarray([(pose.position.x,pose.position.y) for pose in path.poses])
                ax.plot(poses[:,0], poses[:,1], '.',color=clr)
                ax.plot(poses[:,0], poses[:,1], '-',color=clr)

        fig, ax = plt.subplots()
        gridmap.plot(ax)
        plot_path(path, ax, 'r')
        plot_path(path_simple, ax, 'b')
        
        plt.xlabel('x[m]')
        plt.ylabel('y[m]')
        plt.axis('square')
        plt.show()
