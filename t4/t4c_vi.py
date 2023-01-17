#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import math
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('gridmap')
sys.path.append('game')
sys.path.append('player')

import GridMap as gmap
import Game as gm
import Player as pl
from Player import compute_nash

if __name__ == "__main__":
    matrix = np.array([[0.21], [40.21]])
    compute_nash(matrix)

    matrix = np.array([[2, -8, -5],
                       [9, -6, -4],
                       [-9, 3, 4]])
    compute_nash(matrix)

    matrix = np.array([[-2., -9., 9.],
                       [8., 6., - 3.],
                       [5., 4., -4.]])
    compute_nash(matrix)

    matrix = np.array(
        [[-0., - 2.],
         [-3., - 0.],
         [-5., - 2.],
         [6., - 4.],
         [-3., - 1.],
         [-7., - 1.],
         [8., - 3.],
         [-3., 3.]])
    compute_nash(matrix)

    matrix = np.array([[0.21, 40.21],
                       [40.21, 0.21]])
    compute_nash(matrix)

    matrix = np.array([[0.21, 40.21, 0.37],
                       [40.21, 0.21, 0.37]])
    compute_nash(matrix)

    matrix = np.array([[0.21, 40.21, 0.37],
                       [40.21, 0.21, 0.37],
                       [0.21, 0.21, 20.37]])
    compute_nash(matrix)

    # define games:
    #  problem/map name 
    #  game definition file
    games = [("grid", "grid_vi_1"),
             ("grid", "grid_vi_2"),
             ("grid", "grid_vi_3"),
             ("pacman_small", "pacman_small_vi_1"),
             ("pacman_small", "pacman_small_vi_2"),
             ("pacman_small", "pacman_small_vi_3")]

    # fetch individual scenarios
    for game in games:

        # instantiate the map
        gridmap = gmap.GridMap("games/" + game[0] + ".csv")

        # load the game configuration
        players = []
        with open("games/" + game[1] + ".game") as fp:
            for line in fp:
                if line[0] == "#":  # skip comments
                    continue
                q = line.split()

                role = q[0]  # player role
                policy = q[1]  # player policy
                color = q[2]  # player color
                robots = []  # list of player's robots
                for x in range(3, len(q), 2):
                    robots.append((int(q[x]), int(q[x + 1])))

                players.append(pl.Player(robots, role, policy=policy, color=color, game_name=game[0]))

        # instantiate the game
        pursuit_game = gm.Game(gridmap, players)
        pursuit_game.plot_game()
        pursuit_game.plot_pause(4)

        ######################################
        # Simulate the game
        ######################################

        # simulate game of n_steps
        n_steps = 100

        for i in range(0, n_steps):
            # if not all the evaders have been captured
            if not pursuit_game.is_end():
                # make a turn
                pursuit_game.step()

                # plot the game result
                pursuit_game.plot_game()
                pursuit_game.plot_pause(0.01)

        if not pursuit_game.is_end():
            print("Evader escaped")
        else:
            print("GAME OVER")

        # waiting for end
        pursuit_game.plot_pause(2)
