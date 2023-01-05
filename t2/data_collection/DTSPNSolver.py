# -*- coding: utf-8 -*-

import time
import sys
import os
import numpy as np
import math

# import communication messages
from messages import *

from lkh.invoke_LKH import solve_TSP
import dubins


def pose_to_se2(pose):
    return pose.position.x, pose.position.y, pose.orientation.to_Euler()[0]


def se2_to_pose(se2):
    pose = Pose()
    pose.position.x = se2[0]
    pose.position.y = se2[1]
    pose.orientation.from_Euler(se2[2], 0, 0)
    return pose


def configurations_to_path(configurations, turning_radius):
    """
    Compute a closed tour through the given configurations and turning radius, 
    and return densely sampled configurations and length.  

    Parameters
    ----------
    configurations: list Pose
        list of robot configurations in SE3 coordinates (limited to SE2 equivalents), one for each goal
    turning_radius: float
        turning radius for the Dubins vehicle model  

    Returns
    -------
    Path, path_length (float)
        tour as a list of densely sampled robot configurations in SE3 coordinates
    """
    N = len(configurations)
    path = []
    path_len = 0.
    for a in range(N):
        b = (a + 1) % N
        start = pose_to_se2(configurations[a])
        end = pose_to_se2(configurations[b])
        step_size = 0.01 * turning_radius
        dubins_path = dubins.shortest_path(start, end, turning_radius)
        segment_len = dubins_path.path_length()
        step_configurations, _ = dubins_path.sample_many(step_size)
        path = path + [se2_to_pose(sc) for sc in step_configurations]
        path_len += segment_len
    return path, path_len


def create_samples(goals, sensing_radius, position_resolution, heading_resolution):
    """
    Sample the goal regions on the boundary using uniform distribution.

    Parameters
    ----------
    goals: list Vector3
        list of the TSP goal coordinates (x, y, 0)
    sensing_radius: float
        neighborhood of TSP goals  
    position_resolution: int
        number of location at the region's boundary
    heading_resolution: int
        number of heading angles per location
    
    Returns
    -------
    matrix[target_idx][sample_idx] Pose
        2D matrix of configurations in SE3
    """
    samples = []
    for idx, g in enumerate(goals):
        samples.append([])
        for sp in range(position_resolution):
            alpha = sp * 2 * math.pi / position_resolution
            position = [g.x, g.y] + sensing_radius * np.array([math.cos(alpha), math.sin(alpha)])
            for sh in range(heading_resolution):
                heading = sh * 2 * math.pi / heading_resolution
                sample = se2_to_pose((position[0], position[1], heading))
                samples[idx].append(sample)
    return samples


def plan_tour_decoupled(goals, sensing_radius, turning_radius):
    """
    Compute a DTSPN tour using the decoupled approach.  

    Parameters
    ----------
    goals: list Vector3
        list of the TSP goal coordinates (x, y, 0)
    sensing_radius: float
        neighborhood of TSP goals  
    turning_radius: float
        turning radius for the Dubins vehicle model  

    Returns
    -------
    Path, path_length (float)
        tour as a list of robot configurations in SE3 densely sampled
    """

    N = len(goals)

    # find path between each pair of goals (a,b)
    etsp_distances = np.zeros((N, N))
    for a in range(0, N):
        for b in range(0, N):
            g1 = goals[a]
            g2 = goals[b]
            etsp_distances[a][b] = (g1 - g2).norm()

            # Example how to print a small matrix with fixed precision
    # np.set_printoptions(precision=2)
    # print("ETSP distances")
    # print(etsp_distances)

    sequence = solve_TSP(etsp_distances)

    '''
    TODO - homework (Decoupled)
        1) Sample the configurations ih the goal areas (already prepared)
        2) Find the shortest tour
        3) Return the final tour as the points samples (step = 0.01 * radius)
    '''
    position_resolution = heading_resolution = 8
    samples = create_samples(goals, sensing_radius, position_resolution, heading_resolution)

    tour = lio(sequence, samples, turning_radius)

    return configurations_to_path(tour, turning_radius)


def length(tour, radius):
    path = [pose_to_se2(tour_pose) for tour_pose in tour]
    cost = 0.
    for _i in range(len(path)-1):
        dubins_path = dubins.shortest_path(path[_i], path[_i+1], radius)
        cost += dubins_path.path_length()
    return cost


def lio(sequence, samples, radius):
    n = len(samples)
    best_results = []
    for _ in range(100):
        tour = []
        ordered_samples = []
        for idx in sequence:
            j = np.random.randint(len(samples[idx]))
            tour.append(samples[idx][j])
            ordered_samples.append(samples[idx])
        # p_0 = p_n
        tour.insert(0, tour[-1])
        # p_(n+1) = p_1
        tour.append(tour[1])
        cost = length(tour[1:], radius)
        while True:
            # i in [1, ..., n]
            for i in range(1, n + 1):
                current = tour[i]
                current_len = length([tour[i-1], tour[i], tour[i+1]], radius)
                best_len = current_len
                best_sample = current
                for sample_star in ordered_samples[i-1]:
                    if sample_star == current:
                        continue
                    proposal_len = length([tour[i-1], sample_star, tour[i+1]], radius)
                    if proposal_len < best_len:
                        best_len = proposal_len
                        best_sample = sample_star
                tour[i] = best_sample
                if i == 1:
                    tour[-1] = best_sample
                if i == n:
                    tour[0] = best_sample

            new_cost = length(tour[1:], radius)
            if cost - new_cost == 0:
                best_results.append((tour[1:-1], new_cost))
                break
            else:
                cost = new_cost
    s = sorted(best_results, key=lambda x: x[1])
    return s[0][0]


def plan_tour_noon_bean(goals, sensing_radius, turning_radius):
    """
    Compute a DTSPN tour using the NoonBean approach.  

    Parameters
    ----------
    goals: list Vector3
        list of the TSP goal coordinates (x, y, 0)
    sensing_radius: float
        neighborhood of TSP goals  
    turning_radius: float
        turning radius for the Dubins vehicle model  

    Returns
    -------
    Path, path_length (float)
        tour as a list of robot configurations in SE3 densely sampled
    """

    N = len(goals)
    position_resolution = heading_resolution = 8
    samples = create_samples(goals, sensing_radius, position_resolution, heading_resolution)

    # Number of samples per location
    M = position_resolution * heading_resolution
    distances = np.zeros((N * M, N * M))

    '''
    TODO - homework (Noon-Bean approach)
    '''
    # row, column
    prev_max = 0
    for i, j in np.ndindex(distances.shape):
        region1 = i // M
        region2 = j // M
        sample1 = i - region1*M
        sample2 = j - region2*M
        start = samples[region1][sample1]
        end = samples[region2][sample2]
        d = np.nan
        if region1 != region2:
            d = length([start, end], turning_radius)
            prev_max = max(d, prev_max)
        distances[i, j] = d

    distances[np.isnan(distances)] = 4*prev_max
    transformed_distances = np.copy(distances)

    for i, j in np.ndindex(distances.shape):
        region1 = i // M
        region2 = j // M
        if region1 != region2:
            sample2 = (j + 1) % M
            sample2 = region2*M + sample2
            transformed_distances[i, sample2] = distances[i, j] + prev_max

    for region in range(N):
        for vortex in range(M):
            start = region * M + vortex
            next_vortex = (M + vortex + 1) % M
            end = region * M + next_vortex
            transformed_distances[start, end] = 0

    sequence = solve_TSP(transformed_distances)
    prev_region = region = sequence[0] // M
    end_samples = []
    prev_sample = 0
    regions = []
    sample_n = []

    # 64 - 1 = 0 to 63
    # 128 - 1 = 64 to 127
    # 192 - 1 = 128 to 191
    for idx, i in enumerate(sequence):
        region = i // M
        if region != prev_region:
            sample_idx = prev_sample - prev_region * M
            end_samples.append(samples[prev_region][sample_idx])
            regions.append(prev_region)
            sample_n.append(prev_sample)
            prev_region = region
        prev_sample = i
    path, l = configurations_to_path(end_samples, turning_radius)
    return path, l
