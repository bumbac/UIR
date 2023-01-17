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
MIN_NUM_FRONTIERS = 20 - 1


def position_to_coordinates(position, origin, resolution):
    x = (position.x - origin.x) / resolution
    y = (position.y - origin.y) / resolution
    z = (position.z - origin.y) / resolution
    return round(x), round(y)#, round(z)


def find_centroids(data, max_index, grid_map):
    centroids = []
    for i in range(max_index):
        indices = np.argwhere(data == i+1)
        if len(indices) < MIN_NUM_CELLS_FRONTIER and max_index >= MIN_NUM_FRONTIERS:
            continue
        centroids.append(np.mean(indices, axis=0))
    free_edge_centroids = []
    origin_x = grid_map.origin.position.x
    origin_y = grid_map.origin.position.y
    for coord in centroids:
        x_ = coord[1] * grid_map.resolution + origin_x
        y_ = coord[0] * grid_map.resolution + origin_y
        free_edge_centroids.append(Pose(position=Vector3(x_, y_, 0)))
    return free_edge_centroids


def find_clusters(data, max_index, grid_map, laser_scan):
    # equation according to the guidelines in UIR
    f = len(data)
    D = 5
    nr = 1 + round(np.floor(f / D + 0.5))
    # only finding the centroid
    centroids = []
    for i in range(max_index):
        indices = np.argwhere(data == i + 1)
        # filtering frontiers which do not hold much information
        # at the end of the algorithm (small number of frontiers) include all frontiers
        if len(indices) < MIN_NUM_CELLS_FRONTIER and max_index >= MIN_NUM_FRONTIERS:
            continue
        j = 0
        while j + nr < len(indices):
            cluster = indices[j:j+nr]
            centroids.append(np.mean(cluster, axis=0))
            j += nr
        if j < len(indices):
            cluster = indices[j:]
            centroids.append(np.mean(cluster, axis=0))
    multiple_representative_centroids = []
    origin_x = grid_map.origin.position.x
    origin_y = grid_map.origin.position.y
    for coord in centroids:
        x_ = coord[1] * grid_map.resolution + origin_x
        y_ = coord[0] * grid_map.resolution + origin_y
        multiple_representative_centroids.append(Pose(position=Vector3(x_, y_, 0)))
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
        Args:
            grid_map: OccupancyGrid - gridmap for obstacle growing
            start: Pose - robot start pose
            goal: Pose - robot goal pose
        Returns:
            path: Path - path between the start and goal Pose on the map
        """

        if grid_map is None or start is None or goal is None:
            return None

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

            def norm(self, other):
                return np.linalg.norm([self.x - other.x, self.y - other.y])

            def neighbours(self, grid_map):
                width, height = grid_map.width, grid_map.height
                bottom_x = grid_map.origin.position.x / grid_map.resolution
                top_x = bottom_x + width
                bottom_y = grid_map.origin.position.y / grid_map.resolution
                top_y = bottom_y + height

                valid_coords = []
                for x in [-1, 0, 1]:
                    for y in [-1, 0, 1]:
                        if bottom_x <= self.x + x < top_x and bottom_y <= self.y + y < top_y:
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
                # return str(self.h)
                return f"{self.x=}, {self.y=}, {self.flag=}, {self.parent=}, {self.h=}, {self.occupied=}"

            def __eq__(self, other):
                return self.norm(other) <= (robot_size / grid_map.resolution)

            def same(self, other):
                return (self.h + self.price) == (other.h + other.price)

            def __lt__(self, other):
                return (self.h + self.price) < (other.h + other.price)

        start_coordinates = Vector3(round(start.position.x / grid_map.resolution),
                                    round(start.position.y / grid_map.resolution), 0)
        goal_coordinates = Vector3(round(goal.position.x / grid_map.resolution),
                                   round(goal.position.y / grid_map.resolution), 0)
        start_node = Node(start_coordinates, goal_coordinates, None, False)
        goal_node = Node(goal_coordinates, goal_coordinates, None, False)
        grid = [[] for _ in range(grid_map.width)]
        for x in range(grid_map.width):
            for y in range(grid_map.height):
                coordinates = Vector3(
                    round(grid_map.origin.position.x / grid_map.resolution + x),
                    round(grid_map.origin.position.y / grid_map.resolution + y), 0)
                # IMPORTANT, grid_map is row major!
                node = Node(coordinates, goal.position, None, grid_map.data[y][x])
                grid[x].append(node)

        start_x = round(start.position.x / grid_map.resolution + grid_map.width // 2)
        start_y = round(start.position.y / grid_map.resolution + grid_map.height // 2)
        diameter = round(robot_size / grid_map.resolution)
        for i in range(-diameter, diameter):
            for j in range(-diameter, diameter):
                grid[start_x+i][start_y].occupied = False
                grid[start_x + i][start_y + j].occupied = False
                grid[start_x][start_y + j].occupied = False

        goal_x = goal.position.x / grid_map.resolution + grid_map.width // 2
        goal_y = goal.position.y / grid_map.resolution + grid_map.height // 2
        goal_occupied = grid[round(goal_x)][round(goal_y)]
        q = queue.PriorityQueue()
        start_node.price = 0
        start_node.open(None)
        q.put(start_node)
        fallback_node = [float("inf"), None]
        OBSTACLE_SIZE_LIMIT = 1.5 * robot_size / grid_map.resolution
        while not q.empty():
            current_node = q.get()
            if current_node.norm(goal_node) < fallback_node[0]:
                fallback_node = [current_node.norm(goal_node), current_node]
            if current_node.norm(goal_node) < OBSTACLE_SIZE_LIMIT and goal_occupied:
                return True, current_node.get_Path(resolution=grid_map.resolution), current_node.price
            for coord in current_node.neighbours(grid_map):
                x, y = round(coord[0]) + (grid_map.width // 2), round(coord[1]) + (grid_map.height // 2)
                other_node = grid[x][y]
                if not other_node.occupied:
                    if other_node.flag != STATE.CLOSED:
                        distance = current_node.norm(other_node)
                        if (current_node.price + distance) < other_node.price:
                            other_node.open(current_node)
                            q.put(other_node)
        # fallback
        return False, fallback_node[1].get_Path(resolution=grid_map.resolution), fallback_node[1].price

    def relax_goal_accessibility(self, grid_map, path):
        # iterate the path from goal to start
        # until first free cell is found make the occupied cell as free
        it = len(path.poses) - 1
        while it > -1:
            pos = path.poses[it]
            pos = pos.position
            x = round(pos.x / grid_map.resolution + grid_map.width // 2)
            y = round(pos.y / grid_map.resolution + grid_map.height // 2)
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
                    y, x = round(point[0]) + (grid_map.height // 2), round(point[1]) + (grid_map.width // 2)
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

    def find_free_edge_frontiers(self, grid_map, laser_scan, multiple=True):
        """Method to find the free-edge frontiers (edge clusters between the free and unknown areas)
        Args:
            grid_map: OccupancyGrid - gridmap of the environment
            laser_scan: Laser Scan - many variables
        Returns:
            pose_list: Pose[] - list of selected frontiers
        """
        if not grid_map or not laser_scan:
            return None

        a = -1
        c = 10
        mask = np.array([[a] * 3, [a, c, a], [a] * 3])
        data = np.reshape(copy.deepcopy(grid_map.data), (grid_map.height, grid_map.width))
        free_mask = (data < 0.5) * 1
        free_free = ndimage.convolve(free_mask, mask, mode='constant', cval=0.0)
        down_limit = (free_free > 1)
        up_limit = (free_free < 10)
        free_free = np.bitwise_and(down_limit, up_limit)
        a = 1
        c = 0
        mask = np.array([[a] * 3, [a, c, a], [a] * 3])
        data = np.reshape(copy.deepcopy(grid_map.data), (grid_map.height, grid_map.width))
        unknown_mask = (data == 0.5) * 1
        any_unknown = ndimage.convolve(unknown_mask, mask, mode='constant', cval=0.0)
        any_unknown = (any_unknown > 0)
        res = np.bitwise_and(any_unknown, free_free)
        labeled_image, num_labels = skm.label(res, connectivity=2, return_num=True)
        if num_labels < 1:
            return None
        if multiple:
            # assignment F2
            return find_clusters(labeled_image, num_labels, grid_map, laser_scan)
        else:
            # assignment F1
            return find_centroids(labeled_image, num_labels, grid_map)

    def find_inf_frontiers(self, grid_map):
        """Method to find the frontiers based on information theory approach
        Args:
            grid_map: OccupancyGrid - gridmap of the environment
        Returns:
            pose_list: Pose[] - list of selected frontiers
        """

        #TODO:[t1e_expl] find the information rich points in the environment
        return None

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
