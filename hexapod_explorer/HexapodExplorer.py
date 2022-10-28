#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import copy

# import messages
import enum
import queue

import numpy as np
from scipy.ndimage import distance_transform_edt as edt

from messages import *
from enum import Enum

# cpg network
# import cpg.oscilator_network as osc
#


def position_to_coordinates(position, origin, resolution):
    x = np.floor((position.x - origin.x) / resolution)
    y = np.floor((position.y - origin.y) / resolution)
    z = np.floor((position.z - origin.y) / resolution)
    return round(x), round(y), round(z)


class HexapodExplorer:

    def __init__(self):
        pass

    def update_occupied(self, Pmi):
        """method to calculate the Bayesian update of the occupied cell with the current occupancy probability value P_mi
        Args:
            P_mi: float64 - current probability of the cell being occupied
        Returns:
            p_mi: float64 - updated probability of the cell being occupied
        """
        pz_occ = (1 + .95) / 2
        pz_free = 1 - pz_occ
        Pmi_free = 1 - Pmi
        Pmi = (pz_occ*Pmi) / (pz_occ*Pmi + pz_free*Pmi_free)
        if Pmi == 1:
            Pmi = 0.9
        return Pmi

    def update_free(self, Pmi):
        """method to calculate the Bayesian update of the free cell with the current occupancy probability value P_mi
        Args:
            P_mi: float64 - current probability of the cell being occupied
        Returns:
            p_mi: float64 - updated probability of the cell being occupied
        """
        pz_occ = (1 - .95) / 2
        pz_free = 1 - pz_occ
        Pmi_free = 1 - Pmi
        Pmi = (pz_occ * Pmi) / (pz_occ * Pmi + pz_free * Pmi_free)
        if Pmi == 0:
            Pmi = 0.1
        return Pmi

    def bresenham_line(self, start, goal):
        """Bresenham's line algorithm
        Args:
            start: (float64, float64) - start coordinate
            goal: (float64, float64) - goal coordinate
        Returns:
            interlying points between the start and goal coordinate
        """
        (x0, y0) = start[0], start[1]
        (x1, y1) = goal[0], goal[1]
        line = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                line.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                line.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        x = goal[0]
        y = goal[1]
        return line

    def fuse_laser_scan(self, grid_map, laser_scan, odometry):
        """ Method to fuse the laser scan data sampled by the robot with a given 
            odometry into the probabilistic occupancy grid map
        Args:
            grid_map: OccupancyGrid - gridmap to fuse te laser scan to
            laser_scan: LaserScan - laser scan perceived by the robot
            odometry: Odometry - perceived odometry of the robot
        Returns:
            grid_map_update: OccupancyGrid - gridmap updated with the laser scan data
        """
        grid_map_update_object = copy.deepcopy(grid_map)
        if (grid_map is not None) and (odometry is not None) and (laser_scan is not None):
            grid_map_update = grid_map.data.reshape(grid_map_update_object.width, grid_map_update_object.height).T

            origin = grid_map.origin.position
            resolution = grid_map.resolution
            robot_position = odometry.pose.position
            robot_coordinates = position_to_coordinates(robot_position, origin, resolution)
            rotation_matrix = odometry.pose.orientation.to_R()
            free_line_coordinates = copy.deepcopy(laser_scan.distances)

            # calculate pos of obstacle in beam and update free and occupied cells
            for i, beam in enumerate(laser_scan.distances):
                # project the laser scan points to x,y plane with respect to the robot heading
                angle = laser_scan.angle_min + i*laser_scan.angle_increment
                x = beam * np.cos(angle)
                y = beam * np.sin(angle)
                z = 0
                # compensate for the robot odometry
                point = rotation_matrix @ [x, y, z] + [robot_position.x, robot_position.y, z]
                position = Vector3(point[0], point[1], point[2])
                # compensate for the map offset and resolution
                coordinates = position_to_coordinates(position, origin, resolution)
                # raytrace individual scanned points
                free_line_coordinates[i] = \
                    self.bresenham_line((robot_coordinates[0], robot_coordinates[1]), (coordinates[0], coordinates[1]))
                # update occupied (obstacle)
                grid_map_update[coordinates[0]][coordinates[1]] = \
                    self.update_occupied(grid_map_update[coordinates[0]][coordinates[1]])
                # update free
                for _, free_coord in enumerate(free_line_coordinates[i]):
                    x, y = free_coord
                    grid_map_update[x][y] = self.update_free(grid_map_update[x][y])
            # return to 1d array
            grid_map_update = grid_map_update.T
            grid_map_update_object.data = grid_map_update.flatten()
        return grid_map_update_object

    def find_free_edge_frontiers(self, grid_map):
        """Method to find the free-edge frontiers (edge clusters between the free and unknown areas)
        Args:
            grid_map: OccupancyGrid - gridmap of the environment
        Returns:
            pose_list: Pose[] - list of selected frontiers
        """

        #TODO:[t1e_expl] find free-adges and cluster the frontiers
        return None 

    def find_inf_frontiers(self, grid_map):
        """Method to find the frontiers based on information theory approach
        Args:
            grid_map: OccupancyGrid - gridmap of the environment
        Returns:
            pose_list: Pose[] - list of selected frontiers
        """

        #TODO:[t1e_expl] find the information rich points in the environment
        return None

    def grow_obstacles(self, grid_map, robot_size):
        """ Method to grow the obstacles to take into account the robot embodiment
        Args:
            grid_map: OccupancyGrid - gridmap for obstacle growing
            robot_size: float - size of the robot
        Returns:
            grid_map_grow: OccupancyGrid - gridmap with considered robot body embodiment
        # """
        grid_map_grow = copy.deepcopy(grid_map)
        occupied_mask = np.ma.masked_where(grid_map_grow.data >= .5, grid_map_grow.data).mask
        edt_a = edt(~occupied_mask)
        proximity_mask = np.ma.masked_where(edt_a < (robot_size/grid_map.resolution), edt_a).mask
        grid_map_grow.data = np.bitwise_or(proximity_mask, occupied_mask)
        return grid_map_grow


    def plan_path(self, grid_map, start, goal):
        """ Method to plan the path from start to the goal pose on the grid
        Args:
            grid_map: OccupancyGrid - gridmap for obstacle growing
            start: Pose - robot start pose
            goal: Pose - robot goal pose
        Returns:
            path: Path - path between the start and goal Pose on the map
        """

        class STATE(enum.Enum):
            FRESH = 0
            OPEN = 1
            CLOSED = 2

            def __repr__(self):
                if self == STATE.FRESH:
                    return "FRESH"
                if self == STATE.OPEN:
                    return "OPEN"
                return "CLOSED"

        class Node:
            def __init__(self, position, goal_position, parent, occupied):
                self.flag = STATE.FRESH
                self.parent = parent
                self.x = position.x
                self.y = position.y
                self.h = self.heuristic(goal_position)
                self.price = float("inf")
                self.occupied = occupied

            def heuristic(self, other):
                return np.abs(self.x - other.x) + np.abs(self.y - other.y)

            def neighbours(self, width, height):
                valid_coords = []
                for x in [-1, 0, 1]:
                    for y in [-1, 0, 1]:
                        if 0 <= self.x + x < width and 0 <= self.y + y < height:
                            if x == 0 and y == 0:
                                continue
                            valid_coords.append((self.x+x, self.y+y))
                return valid_coords

            def open(self, parent):
                self.parent = parent
                self.flag = STATE.OPEN
                if parent is not None:
                    if self.heuristic(parent) > 1:
                        self.price = parent.price + np.sqrt(2)
                    else:
                        self.price = parent.price + 1

            def close(self):
                self.flag = STATE.CLOSED

            def get_Path(self, resolution=0.1):
                path = Path()
                path.poses.append(Pose(position=Vector3(self.x * resolution, self.y * resolution, 0)))
                parent = self.parent
                while parent is not None:
                    path.poses.append(Pose(position=Vector3(parent.x * resolution, parent.y * resolution, 0)))
                    parent = parent.parent
                path.poses = path.poses[::-1]
                return path

            def __repr__(self):
                return str(self.h)
                # return f"{self.x=}, {self.y=}, {self.flag=}, {self.parent=}, {self.h=}, {self.occupied=}"

            def __eq__(self, other):
                return self.x == other.x and self.y == other.y

            def same(self, other):
                return (self.h + self.price) == (other.h + other.price)

            def __lt__(self, other):
                return (self.h + self.price) < (other.h + other.price)

        if grid_map is None:
            return None
        start_coordinates = Vector3(round(start.position.x / grid_map.resolution),
                                    round(start.position.y / grid_map.resolution), 0)
        goal_coordinates = Vector3(round(goal.position.x / grid_map.resolution),
                                   round(goal.position.y / grid_map.resolution), 0)
        start_node = Node(start_coordinates, goal_coordinates, None, False)
        goal_node = Node(goal_coordinates, goal_coordinates, None, False)
        grid = [[start_node for _1 in range(grid_map.width)] for _ in range(grid_map.height)]
        for x in range(grid_map.width):
            for y in range(grid_map.height):
                coordinates = Vector3(x, y, 0)
                # IMPORTANT, grid_map is row major!
                node = Node(coordinates, goal.position, None, grid_map.data[y][x])
                grid[x][y] = node
        q = queue.PriorityQueue()
        start_node.price = 0
        start_node.open(None)
        q.put(start_node)
        while not q.empty():
            current_node = q.get()
            if current_node == goal_node:
                return current_node.get_Path(resolution=grid_map.resolution)
            for coord in current_node.neighbours(grid_map.width, grid_map.height):
                x, y = round(coord[0]), round(coord[1])
                other_node = grid[x][y]
                if not other_node.occupied:
                    if other_node.flag != STATE.CLOSED:
                        distance = current_node.heuristic(other_node)
                        if distance > 1:
                            distance = np.sqrt(2)
                        if (current_node.price + distance) < other_node.price:
                            other_node.open(current_node)
                            q.put(other_node)
        return None

    def simplify_path(self, grid_map, path):
        """ Method to simplify the found path on the grid
        Args:
            grid_map: OccupancyGrid - gridmap for obstacle growing
            path: Path - path to be simplified
        Returns:
            path_simple: Path - simplified path
        """
        path = copy.deepcopy(path)
        if grid_map == None or path == None:
            return None
        it = 0
        delta = 1
        p = path.poses
        new_path = [p[it]]
        limit = len(p)
        while True:
            if it + delta >= limit:
                break
            else:
                start = (p[it].position.x / grid_map.resolution, p[it].position.y / grid_map.resolution)
                start = round(start[0]), round(start[1])
                goal = (p[it+delta].position.x / grid_map.resolution, p[it+delta].position.y / grid_map.resolution)
                goal = round(goal[0]), round(goal[1])
                line = self.bresenham_line(start, goal)
                obstacle = False
                for i, point in enumerate(line):
                    # IMPORTANT, inverted x and y
                    y, x = round(point[0]), round(point[1])
                    # occupied
                    if grid_map.data[x][y]:
                        obstacle = True
                        new_path.append(p[it+delta-1])
                        it += delta - 1
                        delta = 1
                        break
                if not obstacle:
                    delta += 1
        new_path.append(p[-1])
        path.poses = new_path
        return path

    ###########################################################################
    #INCREMENTAL Planner
    ###########################################################################

    def plan_path_incremental(self, grid_map, start, goal):
        """ Method to plan the path from start to the goal pose on the grid
        Args:
            grid_map: OccupancyGrid - gridmap for obstacle growing
            start: Pose - robot start pose
            goal: Pose - robot goal pose
        Returns:
            path: Path - path between the start and goal Pose on the map
        """

        if not hasattr(self, 'rhs'): #first run of the function
            self.rhs = np.full((grid_map.height, grid_map.width), np.inf)
            self.g = np.full((grid_map.height, grid_map.width), np.inf)

        #TODO:[t1x-dstar] plan the incremental path between the start and the goal Pose
 
        return self.plan_path(grid_map, start, goal), self.rhs.flatten(), self.g.flatten()
