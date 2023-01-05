#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import math
import numpy as np
import scipy.spatial

import collections
import heapq

from dijkstar import Graph, find_path

from environment.Environment import Environment as Environment

# import communication messages
from messages import *

from matplotlib import pyplot as plt


class PRMPlanner:

    def __init__(self, environment, translate_speed, rotate_speed):
        """Initialize the sPRM solver
        Args:  environment: Environment - Map of the environment that provides collision checking
               translate_speed: float - speed of translation motion
               rotate_speed: float - angular speed of rotation motion
        """
        self.environment = environment
        self.translate_speed = translate_speed
        self.rotate_speed = rotate_speed

    # cost function
    def duration(self, p1, p2):
        """Compute duration of movement between two configurations
        Args:  p1: Pose - start pose
               p2: Pose - end pose
        Returns: float - duration in seconds 
        """
        t_translate = (p1.position - p2.position).norm() / self.translate_speed
        t_rotate = p1.orientation.dist(p2.orientation) / self.rotate_speed
        return max(t_translate, t_rotate)

    def create_edge(self, p1, p2, collision_step):
        """Sample an edge between start and goal
        Args:  p1: Pose - start pose
               p2: Pose - end pose
               collision_step: float - minimal time for testing collisions [s]
        Returns: Pose[] - list of sampled poses between the start and goal 
        """
        t = self.duration(p1, p2)
        steps_count = math.ceil(t / collision_step)
        #sample the path
        return slerp_se3(p1, p2, steps_count)   

    def check_collision(self, pth):
        """Check the collision status along a given path
        """
        for se3 in pth:
            if self.environment.check_robot_collision(se3):
                return True
        return False    

    def samples(self, number_of_samples, space):
        limit_x = self.environment.limits_x
        limit_y = self.environment.limits_y
        limit_z = self.environment.limits_z
        yaw = pitch = roll = z = 0
        samples = []
        i = 0
        while i < number_of_samples:
            x = np.random.uniform(*limit_x)
            y = np.random.uniform(*limit_y)
            if "3" in space:
                z = np.random.uniform(*limit_z)
            orientation = Quaternion()
            if "SE" in space:
                yaw = np.random.uniform(0, 2 * np.pi)
                if space == "SE(3)":
                    pitch = np.random.uniform(0, 2 * np.pi)
                    roll = np.random.uniform(0, 2 * np.pi)
                orientation.from_Euler(yaw, pitch, roll)
            position = Vector3(x, y, z)
            sample = Pose(position, orientation)
            if self.environment.check_robot_collision(sample):
                continue
            samples.append(sample)
            i += 1
        return samples

    def plan(self, start, goal, space, number_of_samples, neighborhood_radius, collision_step):
        """Plan the path from start to goal configuration
        Args:  start: Pose - start configuration of the robot in SE(3) coordinates
               goal: Pose - goal configuration of the robot in SE(3) coordinates
               space: String - configuration space type
               number_of_samples: int - number of samples to be generated
               neighborhood_radius: float - neighborhood radius to connect samples [s]
               collision_step: float - minimum step to check collisions [s]
        Returns:  Path - the path between the start and the goal Pose in SE(3) coordinates
                  NavGraph - the navigation graph for visualization of the built roadmap
        """

        # Returned graph
        navgraph = NavGraph()
        # Generated samples
        samples = self.samples(number_of_samples, space)
        # Graph for finding the shortest path (Dijkstra)
        graph = Graph()
        samples.append(start)
        samples.append(goal)
        for i, j in zip(*np.triu_indices(len(samples))):
            if i == j:
                continue
            vs = samples[i]
            ve = samples[j]
            dist = (vs.position - ve.position).norm()
            if dist <= neighborhood_radius:
                edge = self.create_edge(vs, ve, collision_step)
                if self.check_collision(edge):
                    continue
                t = self.duration(vs, ve)
                graph.add_edge(i, j, t)
                graph.add_edge(j, i, t)
        try:
            solution = find_path(graph, len(samples) - 2, len(samples) - 1)
        except:
            print("Dijkstra did not find a solution.") 
            return None, navgraph
        path = Path()
        path.poses.append(start)
        ctr = 0
        navgraph.poses.append(start)
        for i in range(len(solution.nodes) - 1):
            curr_node = samples[solution.nodes[i]]
            next_node = samples[solution.nodes[i+1]]
            edge = self.create_edge(curr_node, next_node, collision_step)
            for e in edge[1:]:
                navgraph.poses.append(e)
                navgraph.edges.append([ctr, ctr+1])
                ctr += 1
                path.poses.append(e)
        return path, navgraph

########################################################
# HELPER functions
########################################################
def slerp_se3(start, end, step_no):
    """Method to compute spherical linear interpolation between se3 poses
    Args:  start: Pose - starting pose
           end: Pose - end pose
           step_no: int - numer of interpolation steps
    Returns: steps: Pose[] - list of the interpolated se3 poses, always at least [start,end]
    """
    #extract the translation
    t1 = start.position
    t2 = end.position
    #extract the rotation
    q1 = start.orientation
    q2 = end.orientation
    #extract the minimum rotation angle
    theta = max(0.01, q1.dist(q2)) / 2 
    if (q1.x*q2.x + q1.y*q2.y + q1.z*q2.z + q1.w*q2.w) < 0:
        q2 = q2 * -1

    steps = []
    if step_no < 2:
        steps = [start,end]
    else:
        for a in np.linspace(0.,1.,step_no+1):
            ta = t1 + (t2-t1)*a
            qa = q1*np.sin( (1-a)*theta ) + q2*np.sin( a*theta )
            qa.normalize()
            pose = Pose(ta, qa)
            steps.append(pose)

    return steps   

