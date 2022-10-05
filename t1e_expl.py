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

#switch to select between the simple variant or the hard variant
SIMPLE_VARIANT = True

if __name__=="__main__":

    explor = explorer.HexapodExplorer()

    dataset = pickle.load( open( "resources/frontier_detection.bag", "rb" ) )

    for gridmap in dataset:
            #find free edges
            points = explor.find_free_edge_frontiers(gridmap)

            #plot the map
            fig, ax = plt.subplots()
            #plot the gridmap
            gridmap.plot(ax)
            #plot the cluster centers
            if points is not None:
                for p in points:
                    plt.plot([p.position.x],[p.position.y],'.', markersize=20)

            plt.xlabel('x[m]')
            plt.ylabel('y[m]')
            ax.set_aspect('equal', 'box')
            plt.show()
