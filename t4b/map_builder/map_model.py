import numpy as np

FREE = 0
WALL = 1
ENTRANCE = 2
GOAL = 3

PATH = "./new_map"

class MapModel:
    def __init__(self, x_size, y_size):
        self.x_size = x_size
        self.y_size = y_size
        self.map = np.zeros((y_size, x_size), dtype=np.uint8)
        self.num_sensors = 0
        self.walls = []
        self.entrances = []
        self.goals = []
        self.sensor_coverage = []
        self.gui = None
        self.mode = "wall"
        self.building_sensor = False

    def set_gui(self, gui):
        self.gui = gui

    def grid_press(self, index):
        if self.mode == "wall":
            if index not in self.walls:
                self.walls.append(index)
                self.map[index[0], index[1]] = WALL
                self.gui.change_color(index, (0.5, 0.5, 0.5, 1))
        if self.mode == "start":
            if index not in self.entrances:
                self.entrances.append(index)
                self.map[index[0], index[1]] = ENTRANCE
                self.gui.change_color(index, (0, 1, 0, 1))
        if self.mode == "goal":
            if index not in self.goals:
                self.goals.append(index)
                self.map[index[0], index[1]] = GOAL
                self.gui.change_color(index, (1, 1, 0, 1))
        if self.building_sensor:
            if index not in self.sensor_coverage[-1]:
                self.sensor_coverage[-1].append(index)
            self.gui.change_color(index, (1, 0, 0, 1))

    def mode_selected(self, mode):
        if self.building_sensor:
            if mode != "sensor":
                return
        self.mode = mode
        if mode == "sensor":
            if not self.building_sensor:
                self.num_sensors += 1
                self.sensor_coverage.append([])
            self.building_sensor = not self.building_sensor
            for sensor in self.sensor_coverage[-1]:
                self.gui.change_color(sensor, (0, 1, 1, 1))
        if mode == "save":
            self.save_and_quit()

    def save_and_quit(self):
        with open(PATH, "w") as file:
            file.write("{} {} {}\n".format(self.x_size, self.y_size, self.num_sensors))
            rotated = np.rot90(self.map)
            for row in rotated:
                for value in row[:-1]:
                    file.write("{} ".format(value))
                file.write("{}\n".format(row[-1]))
            for sensor in self.sensor_coverage:
                for x, y in sensor[:-1]:
                    file.write("{} {} ".format((self.x_size - y - 1), x))
                file.write("{} {}\n".format((self.x_size - sensor[-1][1] - 1), sensor[-1][0]))
