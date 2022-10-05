#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy

# import messages
from messages import *


# cpg network
# import cpg.oscilator_network as osc

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
        pz_occ = (1 + .95 - 0) / 2
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
        Pmi_occ = self.update_occupied(Pmi)
        return 1 - Pmi_occ

    def bresenham_line(self, start, goal):
        """Bresenham's line algorithm
        Args:
            start: (float64, float64) - start coordinate
            goal: (float64, float64) - goal coordinate
        Returns:
            interlying points between the start and goal coordinate
        """
        (x0, y0) = start
        (x1, y1) = goal
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

    def position_to_coordinates(self, position, origin, resolution):
        x = np.floor((position.x - origin.x) / resolution)
        y = np.floor((position.y - origin.y) / resolution)
        z = np.floor((position.z - origin.y) / resolution)
        return int(x), int(y), int(z)


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
        print(grid_map_update_object)
        print(odometry)
        print(laser_scan)
        if (grid_map is not None) and (odometry is not None) and (laser_scan is not None):
            grid_map_update = grid_map.data.reshape(grid_map_update_object.width, grid_map_update_object.height).T

            origin = grid_map.origin.position
            resolution = grid_map.resolution
            robot_position = odometry.pose.position
            robot_coordinates = self.position_to_coordinates(robot_position, origin, resolution)
            rotation_matrix = odometry.pose.orientation.to_R()
            print(rotation_matrix)
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
                coordinates = self.position_to_coordinates(position, origin, resolution)
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
        """

        grid_map_grow = copy.deepcopy(grid_map)

        #TODO:[t1d-plan] grow the obstacles for robot_size

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

        path = Path()
        #add the start pose
        path.poses.append(start)
        
        #TODO:[t1d-plan] plan the path between the start and the goal Pose
        
        #add the goal pose
        path.poses.append(goal)

        return path

    def simplify_path(self, grid_map, path):
        """ Method to simplify the found path on the grid
        Args:
            grid_map: OccupancyGrid - gridmap for obstacle growing
            path: Path - path to be simplified
        Returns:
            path_simple: Path - simplified path
        """
        if grid_map == None or path == None:
            return None
 
        path_simplified = Path()
        #add the start pose
        path_simplified.poses.append(path.poses[0])
        
        #TODO:[t1d-plan] simplifie the planned path
        
        #add the goal pose
        path_simplified.poses.append(path.poses[-1])

        return path_simplified
 
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
