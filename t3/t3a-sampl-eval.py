#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

sys.path.append('environment')
sys.path.append('samplingplanner')
 
import Environment as env
import PRMPlanner as prm

#import communication messages
from messages import *

start1 = Pose(Vector3( 2, 2, 0), Quaternion( 0, 0, 0, 1))
end1   = Pose(Vector3(-2,-2, 0), Quaternion( 0, 0, 0, 1))

start2 = Pose(Vector3( 2,1.5, 0), Quaternion( 0, 0, 0, 1))
end2   = Pose(Vector3( 3, -1, 0), Quaternion( 0, 0, 0, 1))

start3 = Pose(Vector3( 2, 1.5, 0), Quaternion(-0.5,0.5,0.5,0.5))
end3   = Pose(Vector3( 3,-0.5, 0), Quaternion(-0.5,0.5,0.5,0.5))

#define the planning scenarios
scenarios_prm = [
    ("simple_test", start1, end1, 'R2',    [(-5,5), (-5, 5), (0,0)],         500, 1.0, 0.5, 0.2, 1.0),
    ("simple_test", start1, end1, 'SE(2)', [(-5,5), (-5, 5), (0,0)],         800, 1.0, 0.5, 0.2, 1.5),
    ("cubes",       start2, end2, 'R3',    [(1,5), (-2.0,2.0), (-5,0)],      500, 1.0, 0.5, 0.2, 1.0),
    ("cubes",       start3, end3, 'SE(3)', [(1.5,3.5), (-1.0,2.0), (-4,0)], 2000, 1.0, 0.5, 0.2, 1.5),
]

def clean_plot(space):
    plt.clf()
    if space == 'R2' or space == 'SE(2)':
        ax = plt.gca()
        plt.axis('equal')
    else:
        ax = plt.gcf().add_subplot(111, projection='3d')
        ax.view_init(1, 1)
    plt.xlabel("x")
    plt.ylabel("y")
    return ax

#####################################
## EVALUATION OF THE PRM PLANNER
#####################################
for scenario in scenarios_prm:
    name = scenario[0] # name of the scenario (string)
    start = scenario[1] # start configuration (Pose)
    goal = scenario[2] # goal configuration (Pose)
    space = scenario[3] # configuration space
    limits = scenario[4] # limits of translation movement [(xmin,xmax), (ymin,ymax), (zmin,zmax)] all in [m]
    samples = scenario[5] # number of samples to be made by sPRM solver
    translate_speed = scenario[6] # maximum speed of vehicle's translation [m/s]
    rotate_speed = scenario[7] # maximum angular speed of vehicle's rotation [rad/s]
    collision_step = scenario[8] # time step to check possible collisions [s]
    neighborhood_radius = scenario[9] # radius for connecting heighborhoods samples [s]
    
    print("processing scenario: " + name)

    #initiate environment and robot meshes
    environment = env.Environment()
    environment.load_environment("environments/" + name)
    environment.set_limits(limits)
    
    plt.figure(1)
    ax = clean_plot(space)
    environment.plot_environment(ax, start)
    plt.pause(0.1)

    #instantiate the planner
    planner = prm.PRMPlanner(environment, translate_speed, rotate_speed)

    t1 = time.time()
    #plan the path through the environment
    path, navigation_graph = planner.plan(start, goal, space, samples, 
        neighborhood_radius, collision_step)
    t2 = time.time()
    print("Time ", t2-t1, "s")

    if not path == None:
        navigation_graph.plot(ax, plot_edges = True)
        
        #to show the points
        path.plot(ax, style='point', clr=[0,1,0])
        plt.pause(0.1)
        plt.show()

        plt.figure(1)
        ax = clean_plot(space)
        environment.plot_environment(ax, path.poses[0])
        
        for pose in path.poses:
            plt.cla()
            environment.plot_environment(ax, pose)
            path.plot(ax, style='point', clr=[0,1,0])
            plt.pause(0.1)

            if planner.check_collision([pose]):
                print ('fail')
                plt.show()
                break



# sample the space
# in circle connect one vortex to all else in the circle
# check if they are collision free