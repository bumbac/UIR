import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from constants import *


class Gallery:
    def __init__(self, x_size, y_size, num_sensors, walls, entrances, goals, sensor_coverage):
        self.map = np.zeros((y_size, x_size))
        self.x_size = x_size
        self.y_size = y_size
        self.num_sensors = num_sensors
        self.walls = walls
        self.entrances = entrances
        self.goals = goals
        self.sensor_coverage = sensor_coverage

        for wall in walls:
            self.map[wall] = WALL
        for entrance in entrances:
            if not self.entrance_on_the_edge(entrance):
                raise Exception("Entrance must be on the edge of the map!")
            self.map[entrance] = ENTRANCE
        for goal in goals:
            self.map[goal] = GOAL

    # Method that constructs the gallery class from file.
    @classmethod
    def gallery_from_file(cls, file_name):
        walls = []
        entrances = []
        goals = []
        sensor_coverage = []
        with open(file_name, "r") as file:
            y_size, x_size, num_sensors = [int(x) for x in next(file).split()]
            for y_index in range(y_size):
                line = next(file)
                for x_index, square in enumerate([int(x) for x in line.split()]):
                    if square == FREE:
                        continue
                    if square == WALL:
                        walls.append((y_index, x_index))
                        continue
                    if square == ENTRANCE:
                        entrances.append((y_index, x_index))
                        continue
                    if square == GOAL:
                        goals.append((y_index, x_index))
                        continue
                    raise Exception("Impossible number in the file!")
            for sensor_index in range(num_sensors):
                line = next(file)
                sensor_coverage.append([])
                indexes = [int(x) for x in line.split()]
                for i in range(int(len(indexes) / 2)):
                    sensor_coverage[sensor_index].append((indexes[2 * i], indexes[2 * i + 1]))
        return cls(x_size, y_size, num_sensors, walls, entrances, goals, sensor_coverage)
    
    # Plot the path of the attacker
    def plot_path(self, path, show=True, label=None):
        ax = self.plot_gallery(show=False)
        labeled = False
        for i in range(len(path) - 1):
            if label is None or labeled:
                ax.plot([path[i][1], path[i + 1][1]], [path[i][0], path[i + 1][0]], color="k")
            else:
                labeled = True
                ax.plot([path[i][1], path[i + 1][1]], [path[i][0], path[i + 1][0]], color="k", label=round(label, 4))
        if label is not None:
            plt.legend()
        if show:
            plt.show()

    # Plots the gallery map to visualize the wall and sensor layout with entrances and goals.
    def plot_gallery(self, show=True, sensor_prob=None):
        fig, ax = plt.subplots(figsize=(8, 6))
        color_map = colors.ListedColormap(['white', 'gray', 'green', 'yellow'])
        ax.imshow(self.map, cmap=color_map, aspect=1)
        for i, single_sensor_coverage in enumerate(self.sensor_coverage):
            x_offset = (int(i / 4) - 2) / 6
            y_offset = (i % 4 - 2) / 6
            label = ""
            if sensor_prob is not None:
                label = round(sensor_prob[i], 4)
            ax.scatter([y[1] + y_offset for y in single_sensor_coverage], [x[0] + x_offset for x in single_sensor_coverage], label=label)
        ax.set_yticks(np.arange(0, self.y_size, 1))
        ax.set_xticks(np.arange(0, self.x_size, 1))
        ax.set_yticks(np.arange(-.5, self.y_size, 1), minor=True)
        ax.set_xticks(np.arange(-.5, self.x_size, 1), minor=True)
        ax.grid(which='minor', color='k', linestyle='-', linewidth=1)
        ax.xaxis.tick_top()
        if show:
            if sensor_prob is not None:
                plt.legend()
            plt.show()
        return ax

    # Checks for cell being on the edge of the map.
    def entrance_on_the_edge(self, entrance):
        return entrance[0] == 0 or entrance[0] == self.y_size - 1 or entrance[1] == 0 or entrance[1] == self.x_size - 1

    def is_passable(self, cell):
        return self.map[cell] != WALL

    def is_correct_cell(self, cell):
        return 0 <= cell[0] < self.y_size and 0 <= cell[1] < self.x_size
