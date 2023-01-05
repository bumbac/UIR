import numpy as np
from constants import SENSOR_COST, FREE_COST


def pair_evaluation(gallery, path, sensor):
    # passed
    """
    Method to evaluate the value of path-sensor pair

    Parameters
    ----------
    gallery: Gallery
        Map of the environment with information about sensors of a given size
    path: list(tuple(int, int))
        List of coordinates forming the path.
    sensor: int
        Index of the sensor.
    Returns
    -------
    cost: double
        Cost of the path assuming the given sensor is active.
    """
    cost = FREE_COST * len(path)
    for cell in path:
        if cell in gallery.sensor_coverage[sensor]:
            cost += SENSOR_COST
    return cost


def path_evaluation(gallery, path_distribution):
    # passed
    """
    Method to evaluate the cost of path distribution for each sensor

    Parameters
    ----------
    gallery: Gallery
        Map of the environment with information about sensors of given size
    path_distribution: list(list(double), list(tuple(int, int))
        List of two lists. The paths are in the second list, and in the first list are probabilities corresponding to those paths.
    Returns
    -------
    sensor_counts: list(int)
        List of the value of each sensor against given path distribution
    """
    sensor_hits = np.zeros(gallery.num_sensors)
    for idx, prob in enumerate(path_distribution[0]):
        for sensor_idx, sensor_coverage in enumerate(gallery.sensor_coverage):
            for cell in path_distribution[1][idx]:
                sensor_hits[sensor_idx] += FREE_COST * prob
                if cell in sensor_coverage:
                    sensor_hits[sensor_idx] += SENSOR_COST * prob
    return sensor_hits


def best_sensor(gallery, path_distribution):
    # passed
    """
    Method to pick the best sensor against the current path distribution

    Parameters
    ----------
    gallery: Gallery
        Map of the environment with information about sensors of given size
    path_distribution: list(list(double), list(tuple(int, int))
        List of two lists. The paths are in the second list, and in the first list are probabilities corresponding to those paths.
    Returns
    -------
    sensor: int
        Index of the sensor, which achieves the best value against the path distribution.
    reward: double
        Value achieved by the best sensor

    """
    reward = path_evaluation(gallery, path_distribution)
    sensor = np.argmax(reward)
    return sensor, reward[sensor]
