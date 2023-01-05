#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sys
import math
import matplotlib.pyplot as plt

import Player as pl
import GridMap as gmap


class Game:
    def __init__(self, gridmap, players):
        self.gridmap = gridmap  # reference to the map
        self.players = players  # list of players

        self.evaders = self.get_evaders(self.players)
        self.pursuers = self.get_pursuers(self.players)
        self.graveyard = []

        self.trajectories = {}

    def get_evaders(self, players):
        # provide list of not catched evaders
        evaders = []
        for player in players:
            if player.role == pl.EVADER:
                evaders.extend(player.robots)
        return evaders

    def get_pursuers(self, players):
        pursuers = []
        for player in players:
            if player.role == pl.PURSUER:
                pursuers.extend(player.robots)
        return pursuers

    ##################################################
    # Simulation step
    ##################################################
    def step(self):
        # calculate decision for current step
        for i in range(0, len(self.players)):
            pls = self.players[:i] + self.players[i + 1:]
            self.players[i].calculate_step(self.gridmap, self.get_evaders(pls), self.get_pursuers(pls))

        # perform the step
        for player in self.players:
            if not player in self.trajectories:
                self.trajectories[player] = []
                for i in range(len(player.robots)):
                    self.trajectories[player].append([])
                    self.trajectories[player][i].append(player.robots[i])
            player.take_step()
            for i in range(len(player.robots)):
                self.trajectories[player][i].append(player.robots[i])

        self.check_step()

    ##################################################
    # Validity checking function
    ##################################################
    def check_step(self):
        # function to check validity of the performed step
        # valid step for evaders:
        evaders_prev = self.evaders[:]
        pursuers_prev = self.pursuers[:]

        self.evaders = self.get_evaders(self.players)
        self.pursuers = self.get_pursuers(self.players)

        # check for spawning new robots
        assert len(self.evaders) == len(evaders_prev), "cheating!"
        assert len(self.pursuers) == len(pursuers_prev), "cheating!"

        for i in range(0, len(self.evaders)):
            # check for valid step
            assert self.gridmap.dist(evaders_prev[i], self.evaders[i]) == 1, "invalid step"

        for i in range(0, len(self.pursuers)):
            # check for valid steps
            assert self.gridmap.dist(pursuers_prev[i], self.pursuers[i]) == 1, "invalid step"

            # check for evader capture
            for j in range(0, len(self.evaders)):
                # direct catch
                if self.evaders[j] == self.pursuers[i]:
                    # print("Catch", self.evaders[j])
                    self.evader_capture(self.evaders[j])
                    continue
                # catch when cross-passing
                elif self.evaders[j] == pursuers_prev[i] and evaders_prev[j] == self.pursuers[i]:
                    # print("Catch", self.evaders[j])
                    self.evader_capture(self.evaders[j])
                    continue

        self.evaders = self.get_evaders(self.players)
        self.pursuers = self.get_pursuers(self.players)

    def evader_capture(self, pos):
        self.graveyard.append(pos)
        for player in self.players:
            if (player.role == pl.EVADER) and (pos in player.robots):
                player.del_robot(pos)
                # if len(player.robots) == 0:
                # print("player out")
                # self.players.remove(player)

    def is_end(self):
        if len(self.get_evaders(self.players)) == 0:
            return True
        else:
            return False

    ##################################################
    # fun functions
    ##################################################
    def change_status(self):
        print("The hunter becomes prey now!")
        for player in self.players:
            if player.role == pl.EVADER:
                player.role = pl.PURSUER
            elif player.role == pl.PURSUER:
                player.role = pl.EVADER
        self.evaders = self.get_evaders(self.players)
        self.pursuers = self.get_pursuers(self.players)

    ##################################################
    # Helper functions for map plotting
    ##################################################
    def plot_clear(self):
        plt.clf()
        plt.axis('equal')

    def plot_pause(self, time=1):
        plt.pause(time)

    def plot_game(self):
        # method to plot the grid map given as 2D numpy array
        plt.cla()
        ax = plt.gca()
        self.gridmap.plot_map(ax)
        line = ["-", "--"]
        for l, player in zip(line, self.players):
            self.plot_points(player.robots, specs=player.color)
            if player in self.trajectories:
                for traj in self.trajectories[player]:
                    self.plot_trajectory(traj, specs=player.color[0] + l)
        self.plot_points(self.graveyard, 'k+')

    def plot_trajectory(self, traj, specs='r-'):
        x_val = [x[1] for x in traj]
        y_val = [x[0] for x in traj]

        n = len(x_val)
        for i in range(n - 1):
            xa = x_val[i:i + 2]
            ya = y_val[i:i + 2]
            alpha = math.pow((i + 2) / n, 0.9)
            plt.plot(xa, ya, specs, alpha=alpha)

    def plot_points(self, points, specs='r'):
        x_val = [x[1] for x in points]
        y_val = [x[0] for x in points]
        plt.plot(x_val, y_val, specs)

    def save_plot(self, filename):
        plt.gcf().savefig(filename)
