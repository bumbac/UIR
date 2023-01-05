#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import math
import numpy as np

import pickle

import matplotlib.pyplot as plt

class GridMap:
    """ class for representing the game scenario
    """

    def __init__(self, game_name):
        """ 
        """
        self.grid = None
        self.width = None
        self.height = None

        #load the map
        self.grid = np.array(np.genfromtxt(game_name, delimiter=','), dtype=bool)
        #assign width and height
        self.width = self.grid.shape[0]
        self.height = self.grid.shape[1]

        """
        #FLOYD WARSHALL
        self.distance_matrix = {}
        passable = []
        for i in range(self.width):
            for j in range(self.height):
                if self.passable((i, j)):
                    passable.append((i, j))
                    for i_new in range(self.width):
                        for j_new in range(self.height):
                            if self.passable((i_new, j_new)):
                                if i == i_new and j == j_new:
                                    self.distance_matrix[((i, j), (i_new, j_new))] = 0
                                elif (i, j) in self.neighbors4((i_new, j_new)):
                                    self.distance_matrix[((i, j), (i_new, j_new))] = 1
                                else:
                                    self.distance_matrix[((i, j), (i_new, j_new))] = np.inf
        print(passable)
        for k in passable:
            for i in passable:
                for j in passable:
                    self.distance_matrix[(i, j)] = min(self.distance_matrix[(i, j)],
                                                       self.distance_matrix[(i, k)] + self.distance_matrix[(k, j)])
        pickle.dump(self.distance_matrix, open(game_name[:-4] + ".dist", 'wb'))
        """
        #load the distance matrix
        with open(game_name[:-4] + ".dist", "rb") as distance_matrix_file:
            self.distance_matrix = pickle.load(distance_matrix_file)

    ##################################################
    # Planning helper functions
    ##################################################

    def in_bounds(self, coord):
        """ Check the boundaries of the map
        Args:  coord : (int, int) - map coordinate
        Returns: bool - True if the value is inside the grid map, False otherwise
        """
        (x,y) = coord
        return 0 <= x < self.width and 0 <= y < self.height

    def passable(self, coord):
        """ Check the passability of the given cell
        Args:  coord : (int, int) - map coordinate
        Returns: bool - True if the grid map cell is occupied,  False otherwise
        """
        ret = False
        if self.in_bounds(coord):
            (x,y) = coord
            ret = not self.grid[x,y]
        return ret

    def neighbors4(self, coord):
        """ Returns coordinates of passable neighbors of the given cell in 4-neighborhood
        Args:  coord : (int, int) - map coordinate
        Returns: list (int,int) - list of neighbor coordinates 
        """
        (x,y) = coord
        results = [(x+1, y), (x, y-1), (x-1, y), (x, y+1)]
        #results = list(filter(self.in_bounds, results))
        results = list(filter(self.passable, results))
        return results

    def neighbors8(self, coord):
        """ Returns coordinates of passable neighbors of the given cell in 8-neighborhood
        Args:  coord : (int, int) - map coordinate
        Returns: list (int,int) - list of neighbor coordinates 
        """
        (x,y) = coord
        results = [(x+1, y+1), (x+1, y), (x+1, y-1),
                   (x,   y+1),           (x,   y-1),
                   (x-1, y+1), (x-1, y), (x-1, y-1)]
        #results = list(filter(self.in_bounds, results))
        results = list(filter(self.passable, results))
        return results

    def dist(self, start, goal):
        return self.distance_matrix[(start, goal)]

    ##############################################
    ## Helper functions to plot the map
    ##############################################
    def plot_map(self, ax):
        """ Method to plot the occupany grid map
        Args: ax: matplotlib axes object to draw to
        """
        ax.imshow(self.grid, cmap="Greys", interpolation='nearest', vmin=0, vmax=1, origin='lower')

