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
import RRTPlanner as rrt

#import communication messages
from messages import *

start1 = Pose(Vector3( 3, 1, 0), Quaternion( 0, 0, 0, 1))
ends1 = [Pose(Vector3(-3, y, 0), Quaternion( 0, 0, 0, 1)) for y in [-3, 0, 3]]

start2 = Pose(Vector3( 3, 1, 0), Quaternion( 0, 0, 0, 1))
q2 = Quaternion( 0, 0, 1, 1)
q2.normalize()
ends2 = [Pose(Vector3(-3, -1, 0), q2)]

#define the planning scenarios
scenarios_rrt = [
    ("simple_test", start1, ends1, 'R2', [(-5,5), (-5, 5), (0,0)],  300, 1.0, 0.5, 0.2, 1.0, False, 1.0),
    ("simple_test", start1, ends1, 'R2', [(-5,5), (-5, 5), (0,0)], 1000, 1.0, 0.5, 0.2, 1.0, False, 1.0),

    ("simple_test", start1, ends1, 'R2', [(-5,5), (-5, 5), (0,0)],  300, 1.0, 0.5, 0.2, 1.0,  True, 1.0),
    ("simple_test", start1, ends1, 'R2', [(-5,5), (-5, 5), (0,0)], 1000, 1.0, 0.5, 0.2, 1.0,  True, 1.0),

    ("simple_test", start2, ends2, 'SE(2)', [(-5,5), (-5, 5), (0,0)], 1500, 1.0, 0.5, 0.2, 1.5, False, 1.0),
    #('simple_test', , [], 'R2', [(-5, 5), (-5, 5), (0, 0)], 1000, 1.0, 0.5, 0.2, 1.0, True, 1.0)

]

enable_simulation = False

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
for scenario in scenarios_rrt:
    name = scenario[0] # name of the scenario (string)
    start = scenario[1] # start configuration (Pose)
    goals = scenario[2] # list goal configurations (Pose)
    space = scenario[3] # configuration space
    limits = scenario[4] # limits of translation movement [(xmin,xmax), (ymin,ymax), (zmin,zmax)] all in [m]
    samples = scenario[5] # number of samples to be made by sPRM solver
    translate_speed = scenario[6] # maximum speed of vehicle's translation [m/s]
    rotate_speed = scenario[7] # maximum angular speed of vehicle's rotation [rad/s]
    collision_step = scenario[8] # time step to check possible collisions [s]
    neighborhood_radius = scenario[9] # radius for connecting heighborhoods samples [s]
    isrrtstar = scenario[10] # determine RRT* variant: False - RRT, True - RRT*
    steer_step = scenario[11] # step utilized of steering function [s]

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
    planner = rrt.RRTPlanner(environment, translate_speed, rotate_speed)

    t1 = time.time()
    graph = planner.expand_tree(start, space, samples, neighborhood_radius, collision_step, isrrtstar, steer_step)
    t2 = time.time()
    print("Expanding tree: %.3f s" % (t2-t1))
    #print(f"{isrrtstar=}")

    graph.plot(ax, plot_edges = True)

    plt.pause(1)
    # plt.show()

    for goal in goals:
        t1 = time.time()
        path = planner.query(goal, neighborhood_radius, collision_step, isrrtstar)
        t2 = time.time()
        print("Query: %.3f s" % (t2-t1))

        if not path == None:        
            #to show the points
            plt.figure(1)
            ax = clean_plot(space)
            environment.plot_environment(ax, path.poses[0])

            graph.plot(ax, plot_edges = True)
            path.plot(ax, style='point', clr=[0,1,0])
            plt.pause(0.5)
            # plt.show()   
            
            # Simulate the path execution
            if enable_simulation:
                for pose in path.poses:
                    plt.cla()
                    ax = clean_plot(space)
                    environment.plot_environment(ax, pose)
                    path.plot(ax, style='point', clr=[0,1,0])
                    plt.pause(0.1)

                    if planner.check_collision([pose]):
                        print ('fail')
                        plt.show()
                        break
    plt.pause(1)
