import numpy as np
import queue
import copy
from constants import FREE_COST, SENSOR_COST

from dataclasses import dataclass, field
from typing import Any


def cost_function(gallery, sensor_distribution):
    """
    Method to calculate the cost of the cells in the grid given the current sensor distribution

    Parameters
    ----------
    gallery: Gallery
        Map of the environment with information about sensors of given size
    sensor_distribution: list(list(double), list(int))
        List of two lists. In the second list are the sensors, and the first list is the corresponding sensor's probability.
    Returns
    -------
    cost: Matrix of the same size as the map in the gallery
        Matrix filled by cost for each grid cell in the environment. (Costs are set in constants FREE_COST and SENSOR_COST.
        The cost for accessing a cell is FREE_COST + SENSOR_COST * probability of the sensor being active in the cell.)
    """
    matrix = np.full((gallery.y_size, gallery.x_size), FREE_COST)

    for prob, sensor in zip(sensor_distribution[0], sensor_distribution[1]):
        coverage = gallery.sensor_coverage[sensor]
        for cell in coverage:
            matrix[cell] += prob * SENSOR_COST
    return matrix


def best_plan(gallery, cost_matrix):
    """
    Method to calculate the best path to the goal and back given the current cost function

    Parameters
    ----------
    gallery: Gallery
        Map of the environment with information about sensors of a given size
    cost_matrix: Matrix of the same size as the map in the gallery
        Matrix capturing the cost of each cell
    Returns
    -------
    path: list(tuple(int, int))
        List of coordinates visited on the best path
    value: double
        Value of the best path given the cost matrix
    """

    path_there, cost_there = dijkstra(gallery, cost_matrix, gallery.entrances, gallery.goals)
    path_back, cost_back = dijkstra(gallery, cost_matrix, [path_there[-1]], gallery.entrances)
    path_there.extend(path_back[1:])
    cost = cost_there + cost_back - cost_matrix[path_back[0]]
    return path_there, cost


def dijkstra(gallery, cost_matrix, starting_points, end_points) -> (list, int):
    @dataclass(order=True)
    class PItem:
        cost: int
        item: Any = field(compare=False)

    q = queue.PriorityQueue()
    for start in starting_points:
        q.put(PItem(cost_matrix[start], (start, [start])))

    closed = set()

    while True:
        item = q.get()
        cost = item.cost
        cell, path = item.item
        if cell in closed:
            continue
        if cell in end_points:
            break
        closed.add(cell)
        for neighbour in passable_neighbours(cell, gallery):
            new_path = copy.copy(path)
            new_path.append(neighbour)
            q.put(PItem(cost + cost_matrix[neighbour], (neighbour, new_path)))

    return path, cost


def passable_neighbours(cell, gallery):
    neighbours = []
    for move in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        possible_neighbour = (cell[0] + move[0], cell[1] + move[1])
        if gallery.is_correct_cell(possible_neighbour) and gallery.is_passable(possible_neighbour):
            neighbours.append(possible_neighbour)
    return neighbours
