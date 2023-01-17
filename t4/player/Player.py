#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import copy
import time
import random
import numpy as np
from pathlib import Path
import gurobipy as g

import pickle

import GridMap as gmap

PURSUER = 1
EVADER = 2

GREEDY = "GREEDY"
MONTE_CARLO = "MONTE_CARLO"
VALUE_ITERATION = "VALUE_ITERATION"

GAMMA = 0.95

class Player:
    def __init__(self, robots, role, policy=GREEDY, color='r', epsilon=1,
                 timeout=5.0, game_name=None):
        """ constructor of the Player class
        Args: robots: list((in,int)) - coordinates of individual player's robots
              role: PURSUER/EVADER - player's role in the game
              policy: GREEDY/MONTE_CARLO/VALUE_ITERATION - player's policy, 
              color: string - player color for visualization
              epsilon: float - [0,1] epsilon value for greedy policy
              timeout: float - timout for MCTS policy
              game_name: string - name of the currently played game 
        """
        # list of the player's robots
        self.robots = robots[:]
        # next position of the player's robots
        self.next_robots = robots[:]
        self.i2c = None
        self.c2i = None
        if role == "EVADER":
            self.role = EVADER
        elif role == "PURSUER":
            self.role = PURSUER
        else:
            raise ValueError('Unknown player role')

        # selection of the policy
        if policy == GREEDY:
            self.policy = self.greedy_policy
        elif policy == MONTE_CARLO:
            self.policy = self.monte_carlo_policy
            self.timeout = timeout * len(self.robots)  # MCTS planning timeout
            self.tree = {}
            self.max_depth = 10
            self.step = 0
            self.max_steps = 100
            self.beta = 0.95
            self.c = 1
        elif policy == VALUE_ITERATION:
            self.policy = self.value_iteration_policy
            # values for the value iteration policy
            self.loaded_policy = None
            self.gamma = 0.95
        else:
            raise ValueError('Unknown policy')

        # parameters
        self.color = color  # color for plotting purposes
        self.game_name = game_name  # game name for loading vi policies

    #####################################################
    # Game interface functions
    #####################################################
    def add_robot(self, pos):
        """ method to add a robot to the player
        Args: pos: (int,int) - position of the robot
        """
        self.robots.append(pos)
        self.next_robots.append(pos)

    def del_robot(self, pos):
        """ method to remove the player's robot 
        Args: pos: (int,int) - position of the robot to be removed
        """
        idx = self.robots.index(pos)
        self.robots.pop(idx)
        self.next_robots.pop(idx)

    def calculate_step(self, gridmap, evaders, pursuers):
        """ method to calculate the player's next step using selected policy
        Args: gridmap: GridMap - map of the environment
              evaders: list((int,int)) - list of coordinates of evaders in the 
                            game (except the player's robots, if he is evader)
              pursuers: list((int,int)) - list of coordinates of pursuers in 
                       the game (except the player's robots, if he is pursuer)
        """
        self.policy(gridmap, evaders, pursuers)

    def take_step(self):
        """ method to perform the step 
        """
        self.robots = self.next_robots[:]

    #####################################################
    #####################################################
    # GREEDY POLICY
    #####################################################
    #####################################################
    def greedy_policy(self, gridmap, evaders, pursuers, epsilon=1):
        """ Method to calculate the greedy policy action
        Args: gridmap: GridMap - map of the environment
              evaders: list((int,int)) - list of coordinates of evaders in the 
                            game (except the player's robots, if he is evader)
              pursuers: list((int,int)) - list of coordinates of pursuers in 
                       the game (except the player's robots, if he is pursuer)
              epsilon: float (optional) - optional epsilon-greedy parameter
        """
        self.next_robots = self.robots[:]

        # for each of player's robots plan their actions
        for idx in range(0, len(self.robots)):
            robot = self.robots[idx]
            # greedy
            if random.uniform(0, 1) < epsilon:
                if self.role == PURSUER:
                    neighbors = gridmap.neighbors4(robot)
                    d = np.zeros(shape=(len(neighbors), len(evaders)))
                    for i, cell in enumerate(neighbors):
                        for j, e_pos in enumerate(evaders):
                            d[i, j] = gridmap.dist(cell, e_pos)
                    cell_idx, _ = np.unravel_index(d.argmin(), d.shape)
                    self.next_robots[idx] = neighbors[cell_idx]

                if self.role == EVADER:
                    neighbors = gridmap.neighbors4(robot)
                    d = np.zeros(shape=(len(neighbors), len(pursuers)))
                    for i, cell in enumerate(neighbors):
                        for j, p_pos in enumerate(pursuers):
                            d[i, j] = gridmap.dist(cell, p_pos)
                    min_d = -1
                    min_pos = 0
                    for i, cell in enumerate(neighbors):
                        if np.min(d[i]) > min_d:
                            min_d = np.min(d[i])
                            min_pos = cell
                    self.next_robots[idx] = min_pos
            else:
                ##################################################
                # RANDOM Policy
                ##################################################
                # extract possible coordinates to go (actions)
                neighbors = gridmap.neighbors4(robot)
                # introducing randomness in neighbor selection
                random.shuffle(neighbors)

                # select random goal
                self.next_robots[idx] = neighbors[0]
                ##################################################

    #####################################################
    #####################################################
    # VALUE ITERATION POLICY
    #####################################################
    #####################################################
    def init_values(self, gridmap):
        mapping_i2c = {}
        mapping_c2i = {}
        count = 0
        for i in range(gridmap.width):
            for j in range(gridmap.height):
                if gridmap.passable((i, j)):
                    mapping_i2c[count] = (i, j)
                    mapping_c2i[(i, j)] = count
                    count += 1
        return mapping_i2c, mapping_c2i, count

    def random_policy(self, coord_state, gridmap, mapping_c2i, role):
        a, b, c = coord_state
        neigh_a = gridmap.neighbors4(a)
        neigh_b = gridmap.neighbors4(b)
        neigh_c = gridmap.neighbors4(c)
        if role == PURSUER:
            combined_actions = []
            for action_one in neigh_b:
                for action_two in neigh_c:
                    combined_actions.append((mapping_c2i[action_one], mapping_c2i[action_two]))
            return (combined_actions, [1 / len(combined_actions)] * len(combined_actions))
        else:
            combined_actions = []
            for action in neigh_a:
                combined_actions.append(mapping_c2i[action])
            return (combined_actions, [1 / len(combined_actions)] * len(combined_actions))

    def compute_random_policy(self, gridmap):
        mapping_i2c, mapping_c2i, spaces = self.init_values(gridmap)
        values = np.zeros((spaces, spaces, spaces))
        evader_policy = {}
        pursuer_policy = {}
        for a in range(spaces):
            for b in range(spaces):
                for c in range(spaces):
                    coord_state = (mapping_i2c[a], mapping_i2c[b], mapping_i2c[c])
                    evader_policy[(a, b, c)] = self.random_policy(coord_state, gridmap, mapping_c2i, EVADER)
                    pursuer_policy[(a, b, c)] = self.random_policy(coord_state, gridmap, mapping_c2i, PURSUER)
        return values, evader_policy, pursuer_policy, mapping_i2c, mapping_c2i

    def reward(self, state, next_state):
        e, p1, p2 = next_state
        # direct catch
        if e == p1 or e == p2:
            return 1
        prev_e, prev_p1, prev_p2 = state
        # crossing catch
        if prev_e == p1 and prev_p1 == e:
            return 1
        if prev_e == p2 and prev_p2 == e:
            return 1
        return 0


    def compute_vi(self, workload):
        values, evader_policy, pursuer_policy, mapping_i2c, mapping_c2i = workload
        ctr = 0
        games = {}
        old_values = values + 1
        epsilon = 1e-4
        diff = abs(old_values - values) > epsilon
        while diff.any():
            old_values = copy.copy(values)
            ctr += 1
            print(ctr)
            for state in np.ndindex(values.shape):
                # # only statues with difference in value > epsilon
                # if not diff[state]:
                #     continue
                Q = {}
                for evader_action in evader_policy[state][0]:
                    for pursuer_action in pursuer_policy[state][0]:
                        next_state = evader_action, pursuer_action[0], pursuer_action[1]
                        reward = self.reward(state, next_state)
                        Q[next_state] = reward + (1 - reward) * GAMMA * values[next_state]
                matrix = compute_game_matrix(Q)
                games[state] = matrix
                game_value, _ = compute_nash(matrix)
                values[state] = game_value
            diff = abs(old_values - values) > epsilon

        for state in np.ndindex(values.shape):
            matrix = games[state]
            pursuer_strategy = maximin(matrix)
            evader_strategy = maximin(matrix.T, minimize=True)
            evader_policy[state] = evader_policy[state][0], evader_strategy
            pursuer_policy[state] = pursuer_policy[state][0], pursuer_strategy

        return values, evader_policy, pursuer_policy, mapping_i2c, mapping_c2i

    def value_iteration_policy(self, gridmap, evaders, pursuers):
        """
        Method to calculate the value-iteration policy action

        Parameters
        ----------
        gridmap: GridMap
            Map of the environment
        evaders: list((int,int))
            list of coordinates of evaders in the game (except the player's robots, if he is evader)
        pursuers: list((int,int))
            list of coordinates of pursuers in the game (except the player's robots, if he is pursuer)
        """
        self.next_robots = self.robots[:]

        # if there are not precalculated values for policy
        if not self.loaded_policy:
            policy_file = Path("policies/" + self.game_name + ".policy")
            ###################################################
            # if there is policy file, load it...
            ###################################################
            if policy_file.is_file():
                # load the strategy file
                self.loaded_policy = pickle.load(open(policy_file, 'rb'))
            ###################################################
            # ...else calculate the policy
            ###################################################
            else:
                workload = self.compute_random_policy(gridmap)
                values, evader_policy, pursuer_policy, mapping_i2c, mapping_c2i = self.compute_vi(workload)
                self.loaded_policy = (values, evader_policy, pursuer_policy, mapping_i2c, mapping_c2i)

                pickle.dump(self.loaded_policy, open(policy_file, 'wb'))

        values, evader_policy, pursuer_policy, mapping_i2c, mapping_c2i = self.loaded_policy

        if self.role == PURSUER:
            state = (mapping_c2i[evaders[0]], mapping_c2i[self.robots[0]], mapping_c2i[self.robots[1]])
        else:
            state = (mapping_c2i[self.robots[0]], mapping_c2i[pursuers[0]], mapping_c2i[pursuers[1]])

        if self.role == PURSUER:
            action_index = np.random.choice(tuple(range(len(pursuer_policy[state][0]))), p=pursuer_policy[state][1])
            action = pursuer_policy[state][0][action_index]
            self.next_robots[0] = mapping_i2c[action[0]]
            self.next_robots[1] = mapping_i2c[action[1]]
        else:
            action_index = np.random.choice(tuple(range(len(evader_policy[state][0]))), p=evader_policy[state][1])
            action = evader_policy[state][0][action_index]
            self.next_robots[0] = mapping_i2c[action]
            #####################################################


def compute_game_matrix(Q):
    actions = list(Q.keys())
    e_actions = [a[0] for a in actions]
    p_actions = [(a[1], a[2]) for a in actions]
    e_map = {}
    p_map = {}
    e_ctr = 0
    p_ctr = 0
    for e, p in zip(e_actions, p_actions):
        if e not in e_map:
            e_map[e] = e_ctr
            e_ctr += 1
        if p not in p_map:
            p_map[p] = p_ctr
            p_ctr += 1
    M = len(p_map)
    N = len(e_map)
    matrix = np.zeros((M, N))
    for e, p in zip(e_actions, p_actions):
        i = p_map[p]
        j = e_map[e]
        matrix[i, j] = Q[(e, p[0], p[1])]
    return matrix


def maximin(matrix, minimize=False):
    """
    Method to calculate the value-iteration policy action

    Parameters
    ----------
    matrix: n times m array of floats
        Game utility matrix

    Returns
    -------
    value:float
        computed value of the game
    strategy:float[n]
        probability of player 1 playing each action in nash equilibrium
    """
    min_val = np.min(matrix)
    if min_val < 0:
        matrix = matrix + abs(min_val)
    # compute Nash Equilibrium
    model = g.Model()
    model.setParam("OutputFlag", 0)
    U = model.addVar(name="U")
    # M is number of rows
    # N is number of columns
    M, N = matrix.shape
    path_vars = []
    for i in range(M):
        path_vars.append(model.addVar(name="Path " + str(i)))
    if minimize:
        model.setObjective(U, g.GRB.MINIMIZE)
        for j in range(N):
            model.addConstr(g.quicksum(
                [matrix[i, j] * path_vars[i] for i in range(M)]
            ) <= U)
    else:
        model.setObjective(U, g.GRB.MAXIMIZE)
        for j in range(N):
            model.addConstr(g.quicksum(
                [matrix[i, j] * path_vars[i] for i in range(M)]
            ) >= U)
    model.addConstr(g.quicksum(path_vars) == 1)
    for pv in path_vars:
        model.addConstr(pv >= 0)
    model.optimize()
    if model.status == 4:
        model.setParam("DualReductions", 0)
        model.reset()
        model.optimize()
    M_strategy = np.array([var.X for var in path_vars])
    return M_strategy


def compute_nash(matrix):
    M_strategy = maximin(matrix)
    N_strategy = maximin(matrix.T, minimize=True)
    value = M_strategy.T @ matrix @ N_strategy
    return value, M_strategy
