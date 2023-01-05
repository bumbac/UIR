#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import math
import numpy as np
import scipy.spatial

import collections
import heapq
# from environment.Environment import Environment as Environment
import Environment as env

# import communication messages
from messages import *

from matplotlib import pyplot as plt


class RRRTNode:
    def __init__(self, parent, pose, cost, id):
        self.parent = parent
        self.pose = pose
        self.cost = cost
        self.id = id
        self.head = []


class RRTPlanner:
    def __init__(self, environment, translate_speed, rotate_speed):
        """Initialize the sPRM solver
        Args:  environment: Environment - Map of the environment that provides collision checking
               translate_speed: float - speed of translation motion
               rotate_speed: float - angular speed of rotation motion
        """
        self.environment = environment
        self.translate_speed = translate_speed
        self.rotate_speed = rotate_speed
        self.nodes = []

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
        if steps_count <= 1:
            return [p1, p2]
        else:
            parameters = np.linspace(0., 1., steps_count + 1)
            return slerp_se3(p1, p2, parameters)

    def check_collision_edges(self, pth, collision_step):
        return self.check_collision(self.create_edge(pth[0], pth[1], collision_step))

    def check_collision(self, pth):
        """Check the collision status along a given path
        """
        for se3 in pth:
            if self.environment.check_robot_collision(se3):
                return True
        return False

    def Steer(self, x_nearest, x_rand, steer_step):
        """Steer function of RRT algorithm 
        Args:  x_nearest: Pose - pose from which the tree expands
               x_rand: Pose - random pose towards which the tree expands
               steer_step: float - maximum distance from x_nearest [s]
        Returns: Pose - new pose to be inserted into the tree
        """
        t = self.duration(x_nearest, x_rand)
        if t < steer_step:
            return x_rand
        else:
            parameter = steer_step / t
            return slerp_se3(x_nearest, x_rand, [parameter])[0]

    def samples(self, number_of_samples=100, space="R2"):
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

    def expand_tree(self, start, space, number_of_samples, neighborhood_radius, collision_step,
                    isrrtstar, steer_step):
        """Expand the RRT(*) tree for finding the shortest path
        Args:  start: Pose - start configuration of the robot in SE(3) coordinates
               space: String - configuration space type
               number_of_samples: int - number of samples to be generated
               neighborhood_radius: float - neighborhood radius to connect samples [s]
               collision_step: float - minimum step to check collisions [s]
               isrrtstar: bool - determine RRT* variant
               steer_step: float - step utilized of steering function
        Returns:  NavGraph - the navigation graph for visualization of the built roadmap
        """

        # steer is limiting by circle, instead of reaching the desired node, select the one on the circle border
        root = RRRTNode(None, start, 0.0, 0)
        # RRT class
        self.nodes = [root]
        # Pose class
        samples = []
        max_n = 0
        sample_idx = 0
        id = 1
        navgraph = NavGraph()
        navgraph.poses.append(root.pose)
        while max_n < number_of_samples:
            if sample_idx == len(samples):
                samples = self.samples(number_of_samples // 2, space)
                sample_idx = 0
            x_rand = samples[sample_idx]
            sample_idx += 1
            # nearest is a list of nodes
            nearest = sorted([(start, self.duration(start.pose, x_rand)) for start in self.nodes], key=lambda x: x[1])
            x_min = nearest[0][0]
            x_new = self.Steer(x_min.pose, x_rand, steer_step)
            if not self.check_collision_edges([x_min.pose, x_new], collision_step):
                max_n += 1
                c_min = self.duration(x_min.pose, x_new)
                X_near = []
                if isrrtstar:
                    for node in self.nodes:
                        duration = self.duration(node.pose, x_new)
                        if duration < neighborhood_radius:
                            X_near.append((node, duration))
                    for x_near, duration in X_near:
                        if not self.check_collision_edges([x_near.pose, x_new], collision_step) \
                                and (x_near.cost + duration) < (x_min.cost + c_min):
                            x_min = x_near
                            c_min = duration

                # nearest=parent=node, new_sample=Pose, cost=parent.cost + t, id=id
                new_node = RRRTNode(x_min, x_new, x_min.cost + c_min, id)
                id += 1
                self.nodes.append(new_node)
                x_min.head.append(new_node)
                navgraph.poses.append(new_node.pose)
                navgraph.edges.append((x_min.id, new_node.id))
                if isrrtstar:
                    for x_near, _ in X_near:
                        duration = self.duration(new_node.pose, x_near.pose)
                        if not self.check_collision_edges([new_node.pose, x_near.pose], collision_step) \
                                and (new_node.cost + duration) < x_near.cost:
                            x_parent = x_near.parent
                            if (x_parent.id, x_near.id) in navgraph.edges:
                                idx = navgraph.edges.index((x_parent.id, x_near.id))
                                navgraph.edges.pop(idx)
                                x_parent.head.remove(x_near)
                                new_node.head.append(x_near)
                                x_near.parent = new_node
                                delta = (x_near.cost - new_node.cost - duration)
                                x_near.cost = new_node.cost + duration
                                self.update_cost(x_near, delta)
                                if self.check_collision_edges([new_node.pose, x_near.pose], collision_step):
                                    print("PROBLEM")
                                navgraph.edges.append((new_node.id, x_near.id))
        return navgraph

    def update_cost(self, node, delta):
        for child in node.head:
            child.cost -= delta
            self.update_cost(child, delta)

    def query(self, goal, neighborhood_radius, collision_step, isrrtstar):
        """Retrieve path for the goal configuration
        Args:  goal: Pose - goal configuration of the robot in SE(3) coordinates
               neighborhood_radius: float - neighborhood radius to connect samples [s]
               collision_step: float - minimum step to check collisions [s]
               isrrtstar: bool - determine RRT* variant
        Returns:  Path - the path between the start and the goal Pose in SE(3) coordinates
        """
        path = Path()
        s = sorted([(start, self.duration(start.pose, goal)) for start in self.nodes], key=lambda x: x[1])
        node_iter = iter(s)
        x_min = next(node_iter)[0]
        while self.check_collision_edges([x_min.pose, goal], collision_step):
            x_min = next(node_iter)[0]
        reversed_path = [goal]
        while x_min.parent:
            reversed_path.append(x_min.pose)
            x_min = x_min.parent
        start = x_min.pose
        for node in reversed_path[::-1]:
            edge = self.create_edge(start, node, collision_step)
            for p in edge:
                path.poses.append(p)
            start = node

        return path


########################################################
# HELPER functions
########################################################
def slerp_se3(start, end, parameters):
    """Method to compute spherical linear interpolation between se3 poses
    Args:  start: Pose - starting pose
           end: Pose - end pose
           parameters: float[] - array of parameter in (0,1), 0-start, 1-end
    Returns: steps: Pose[] - list of the interpolated se3 poses, always at least [start,end]
    """
    # extract the translation
    t1 = start.position
    t2 = end.position
    # extract the rotation
    q1 = start.orientation
    q2 = end.orientation
    # extract the minimum rotation angle
    theta = max(0.01, q1.dist(q2)) / 2
    if (q1.x * q2.x + q1.y * q2.y + q1.z * q2.z + q1.w * q2.w) < 0:
        q2 = q2 * -1

    steps = []
    for a in parameters:
        ta = t1 + (t2 - t1) * a
        qa = q1 * np.sin((1 - a) * theta) + q2 * np.sin(a * theta)
        qa.normalize()
        pose = Pose(ta, qa)
        steps.append(pose)

    return steps
