#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import copy

# import messages
import enum
import queue


import numpy as np
import scipy.ndimage as ndimage
import skimage.measure as skm

from sklearn.cluster import KMeans

from messages import *

MIN_NUM_CELLS_FRONTIER = 4
MIN_NUM_FRONTIERS = 0


def position_to_coordinates(position, origin, resolution):
    # calculate discretized X, Y coordinates from 3D position
    x = (position.x - origin.x) / resolution
    y = (position.y - origin.y) / resolution
    # z = (position.z - origin.y) / resolution
    return round(x), round(y)#, round(z)


def coordinates_to_position(coordinates, origin, resolution):
    # calculate 3D position from discretized coordinates
    x = coordinates[0] * resolution + origin.x
    y = coordinates[1] * resolution + origin.y
    z = 0
    return Vector3(x, y, z)


def find_centroids(data, max_index, grid_map):
    # find centroid of all points in one frontier
    centroids = []
    for i in range(max_index):
        indices = np.argwhere(data == i+1)
        if len(indices) < MIN_NUM_CELLS_FRONTIER:
            continue
        centroids.append(np.mean(indices, axis=0))
    free_edge_centroids = []
    for coord in centroids:
        position = coordinates_to_position((coord[1], coord[0]), grid_map.origin.position,
                                           grid_map.resolution)
        free_edge_centroids.append(Pose(position=position))
    return free_edge_centroids


def find_clusters(data, max_index, grid_map, laser_scan):
    # equation according to the guidelines in UIR
    f = len(data)
    D = laser_scan.range_max / grid_map.resolution
    nr = 1 + round(np.floor(f / D + 0.5))
    # only finding the centroid
    centroids = []
    model = KMeans(n_clusters=nr)
    for i in range(max_index):
        # cell indices in the grow obstacles grid
        indices = np.argwhere(data == i + 1)
        # filter frontiers which are located in noisy area (only MIN_NUM_CELLS_FRONTIER in cluster)
        if len(indices) < MIN_NUM_CELLS_FRONTIER:
            continue
        model.fit(indices)
        centroids.extend(model.cluster_centers_)
    multiple_representative_centroids = []
    for coord in centroids:
        # convert centroids in gridmap coordinates to position in 3D
        position = coordinates_to_position((coord[1], coord[0]), grid_map.origin.position,
                                           grid_map.resolution)
        multiple_representative_centroids.append(Pose(position=position))
    return multiple_representative_centroids


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
        if Pmi > 0.95:
            Pmi = 0.95
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
        if Pmi < 0.05:
            Pmi = 0.05
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
            grid_map_update = grid_map_update_object.data.T
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
                    self.bresenham_line(robot_coordinates, coordinates)
                # update occupied (obstacle)
                grid_map_update[coordinates] = \
                    self.update_occupied(grid_map_update[coordinates])
                # update free
                for _, free_coord in enumerate(free_line_coordinates[i]):
                    x, y = free_coord
                    grid_map_update[x][y] = self.update_free(grid_map_update[x][y])
            # return to 1d array
            grid_map_update_object.data = grid_map_update.T
        return grid_map_update_object

    def grow_obstacles(self, grid_map, robot_size):
        """ Method to grow the obstacles to take into account the robot embodiment
        Args:
            grid_map: OccupancyGrid - gridmap for obstacle growing
            robot_size: float - size of the robot
        Returns:
            grid_map_grow: OccupancyGrid - gridmap with considered robot body embodiment
        # """
        if grid_map is None:
            return grid_map
        grid_map_grow = copy.deepcopy(grid_map)
        # mask of obstacles and unknown areas
        occupied_mask = np.ma.masked_where(grid_map_grow.data >= .5, grid_map_grow.data).mask
        edt_a = ndimage.distance_transform_edt(~occupied_mask)
        # mask of distances to obstacle or uknown area lesser than robot_size
        proximity_mask = np.ma.masked_where(edt_a < (robot_size / grid_map.resolution), edt_a).mask
        grid_map_grow.data = np.bitwise_or(proximity_mask, occupied_mask)
        return grid_map_grow

    def plan_path(self, grid_map, start, goal, robot_size):
        """ Method to plan the path from start to the goal pose on the grid
            A star with manhattan heuristic
        Args:
            grid_map: OccupancyGrid - gridmap for obstacle growing
            start: Pose - robot start pose
            goal: Pose - robot goal pose
        Returns:
            path: Path - path between the start and goal Pose on the map
        """

        if grid_map is None or start is None or goal is None:
            return None
        resolution = grid_map.resolution
        origin = grid_map.origin.position
        DIAGONAL_MOVEMENT = np.sqrt(2)

        class Node:
            def __init__(self, position, goal_position, parent, occupied):
                self.position = position
                self.x, self.y = position_to_coordinates(position, origin, resolution)
                self.h = self.heuristic(position_to_coordinates(goal_position, origin, resolution))
                self.price = float("inf")
                self.occupied = occupied
                self.closed = False
                self.parent = parent

            def heuristic(self, coordinates):
                x, y = coordinates
                return np.abs(self.x - x) + np.abs(self.y - y)

            def distance(self, other):
                dist = np.abs(self.x - other.x) + np.abs(self.y - other.y)
                if dist > 1:
                    return DIAGONAL_MOVEMENT
                return dist

            def norm(self, other):
                return np.linalg.norm([self.position.x - other.position.x, self.position.y - other.position.y])

            def neighbours(self):
                width, height = grid_map.width, grid_map.height
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
                if parent is not None:
                    dist = self.distance(parent)
                    self.price = parent.price + dist

            def close(self):
                self.closed = True

            def get_Path(self):
                path = Path()
                parent = self
                while parent is not None:
                    path.poses.append(Pose(position=parent.position))
                    parent = parent.parent
                path.poses = path.poses[::-1]
                return path

            def __lt__(self, other):
                return (self.h + self.price) < (other.h + other.price)

        start_node = Node(start.position, goal.position, None, False)
        goal_node = Node(goal.position, goal.position, None, False)
        grid = [[] for _ in range(grid_map.width)]
        for x in range(grid_map.width):
            for y in range(grid_map.height):
                position = coordinates_to_position((x, y), origin, resolution)
                # IMPORTANT, grid_map is row major!
                node = Node(position, goal.position, None, grid_map.data[y][x])
                grid[x].append(node)

        # set to free nodes diameter around the robot
        # which is surely accessible because robot is there
        diameter = round(robot_size / resolution)
        for i in range(-diameter, diameter):
            for j in range(-diameter, diameter):
                x_ = start_node.x + i
                y_ = start_node.y + j
                # check bounds
                if 0 <= x_ <= grid_map.width:
                    grid[x_][start_node.y].occupied = False
                    if 0 <= y_ <= grid_map.height:
                        grid[x_][y_].occupied = False
                if 0 <= y_ <= grid_map.height:
                    grid[start_node.x][y_].occupied = False

        q = queue.PriorityQueue()
        start_node.price = 0
        start_node.open(None)
        q.put(start_node)
        fallback_node = [float("inf"), None]
        OBSTACLE_SIZE_LIMIT = 1.5 * robot_size
        SUCCESS = True
        FALLBACK = False
        while not q.empty():
            current_node = q.get()
            dist = current_node.norm(goal_node)
            if dist < fallback_node[0]:
                fallback_node = [dist, current_node]
            if dist < OBSTACLE_SIZE_LIMIT:
                return SUCCESS, current_node.get_Path(), current_node.price * resolution
            for coord in current_node.neighbours():
                x, y = coord
                other_node = grid[x][y]
                if not other_node.occupied:
                    if not other_node.closed:
                        distance = current_node.distance(other_node)
                        if (current_node.price + distance) < other_node.price:
                            other_node.open(current_node)
                            q.put(other_node)
        # fallback
        return FALLBACK, fallback_node[1].get_Path(), fallback_node[1].price * resolution

    def relax_goal_accessibility(self, grid_map, path):
        # iterate the path from goal to start
        # until first free cell is found make the occupied cell as free
        it = len(path.poses) - 1
        while it > -1:
            pos = path.poses[it]
            x, y = position_to_coordinates(pos.position, grid_map.origin.position, grid_map.resolution)
            if grid_map.data[y][x]:
                grid_map.data[y][x] = False
            it -= 1
        return grid_map

    def simplify_path(self, grid_map, path):
        """ Method to simplify the found path on the grid
        Args:
            grid_map: OccupancyGrid - gridmap for obstacle growing
            path: Path - path to be simplified
        Returns:
            path_simple: Path - simplified path
        """
        if grid_map is None or path is None:
            return None
        p = path.poses
        if len(p) == 0:
            return path
        path = copy.deepcopy(path)
        limit = len(p)
        it = 0
        delta = 1
        new_path = [p[it]]
        # create a path of accessible cells to the goal
        grid_map = self.relax_goal_accessibility(grid_map, path)
        # iterate the raycast until an obstacle is found
        while True:
            if it + delta >= limit:
                break
            else:
                start = position_to_coordinates(p[it].position, grid_map.origin.position, grid_map.resolution)
                goal = position_to_coordinates(p[it+delta].position, grid_map.origin.position, grid_map.resolution)
                line = self.bresenham_line(start, goal)
                obstacle = False
                for i, point in enumerate(line):
                    # IMPORTANT, inverted x and y
                    y, x = round(point[0]), round(point[1])
                    if grid_map.data[x][y]:
                        # occupied
                        # do not use the raycast from the it position
                        # iterate one cell further and raycast again
                        obstacle = True
                        new_path.append(p[it+delta-1])
                        it += delta - 1
                        delta = 1
                        break
                if not obstacle:
                    # you can raycast because no obstacle was found in it
                    delta += 1
        new_path.append(p[-1])
        path.poses = new_path
        dist = 0
        prev_pose = path.poses[0]
        for pose in path.poses:
            dist += np.linalg.norm([pose.position.x - prev_pose.position.x,
            pose.position.y - prev_pose.position.y])
            prev_pose = pose

        return path, dist

    def find_free_edge_frontiers(self, grid_map, laser_scan, multiple=True):
        """Method to find the free-edge frontiers (edge clusters between the free and unknown areas)
        Args:
            grid_map: OccupancyGrid - gridmap of the environment
            laser_scan: Laser Scan - many variables
            multiple: bool - calculate multiple-representative free-edge cluster frontiers (f2 assignment)
        Returns:
            pose_list: Pose[] - list of selected frontiers
        """
        if not grid_map or not laser_scan:
            return None

        # detection mask for unoccupied cell with at least one unoccupied neighbour cell
        a = -1
        # center value
        c = 10
        mask = np.array([[a] * 3, [a, c, a], [a] * 3])
        data = np.reshape(copy.deepcopy(grid_map.data), (grid_map.height, grid_map.width))
        # mask unoccupied cells as true
        free_mask = (data < 0.5) * 1
        # calculate value using mask
        free_free = ndimage.convolve(free_mask, mask, mode='constant', cval=0.0)
        # cells with value > 1 are unoccupied in the center
        down_limit = (free_free > 1)
        # cells with value < 10 have at least one unoccupied neighbour cell
        up_limit = (free_free < 10)
        # if center cell is unoccupied and has at least one unoccupied neighbour cell accept it
        # as free cell (not yet frontier)
        free_free = np.bitwise_and(down_limit, up_limit)

        # detection mask for unknown cells in the neighbourhood
        a = 1
        c = 0
        mask = np.array([[a] * 3, [a, c, a], [a] * 3])
        data = np.reshape(copy.deepcopy(grid_map.data), (grid_map.height, grid_map.width))
        unknown_mask = (data == 0.5) * 1
        any_unknown = ndimage.convolve(unknown_mask, mask, mode='constant', cval=0.0)
        # at least one unknown cell in the neighbourhood
        any_unknown = (any_unknown > 0)
        # cell which is unoccupied in the middle and has at least one unoccupied and one uknown cell
        # in the neighbourhood
        res = np.bitwise_and(any_unknown, free_free)
        labeled_image, num_labels = skm.label(res, connectivity=2, return_num=True)
        if num_labels < 1:
            # did not find any frontiers
            return None
        if multiple:
            # assignment F2
            return find_clusters(labeled_image, num_labels, grid_map, laser_scan)
        else:
            # assignment F1
            return find_centroids(labeled_image, num_labels, grid_map)
