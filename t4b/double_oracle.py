import numpy as np
from sensor_oracle import best_sensor, pair_evaluation
from planning_oracle import best_plan, cost_function
import gurobipy as g


def double_oracle(gallery, epsilon=1e-6):
    """
    Method to compute optimal strategy for attacker and defender using double oracle algorithm and oracles implemented in previous steps

    Parameters
    ----------
    gallery: Gallery
        Map of the environment with information about sensors of a given size
    epsilon: double
        The distance between both player's best response values required to terminate the algorithm
    Returns
    -------
    sensor_strategy: list(double)
        Optimal strategy as a probability distribution over sensors
    sensors: list(int)
        List of sensors used as actions
    path_strategy: list(double)
        Optimal strategy as a probability distribution over paths
    paths: list(list(tuple(int, int)))
        List of all the paths used as actions
    """
    # pick action for adversary
    sensor_idx = 0
    sensor_distribution = [[1.0], [sensor_idx]]
    # pick action for planner
    path_distribution = [[], []]
    path, path_cost = best_plan(gallery, cost_function(gallery, sensor_distribution))
    path_distribution[0].append(1.0)
    path_distribution[1].append(path)
    value = pair_evaluation(gallery, path, sensor_idx)
    game_matrix = np.zeros((1, 1))
    game_matrix[0, 0] = value

    while True:
        # compute Nash Equilibrium
        model = g.Model()
        model.setParam("OutputFlag", 0)
        U = model.addVar(name="U")
        path_vars = []
        # M is number of rows == paths
        # N is number of columns == sensors
        M, N = game_matrix.shape
        for i in range(M):
            path_vars.append(model.addVar(lb=0, ub=1.0, vtype=g.GRB.CONTINUOUS, name="Path " + str(i)))
        model.setObjective(U, g.GRB.MINIMIZE)
        for j in range(N):
            model.addConstr(g.quicksum(
                [game_matrix[i, j] * path_vars[i] for i in range(M)]
            ) <= U, "cons1")
        model.addConstr(g.quicksum(path_vars) == 1, "cons2")
        for pv in path_vars:
            model.addConstr(pv >= 0, "cons3")
        model.optimize()
        path_distribution[0] = [var.X for var in path_vars]
        # compute Nash Equilibrium
        model = g.Model()
        model.setParam("OutputFlag", 0)
        U = model.addVar(name="U")
        sensor_path_vars = []
        for j in range(N):
            sensor_path_vars.append(model.addVar(lb=0, ub=1.0, vtype=g.GRB.CONTINUOUS, name="Sensor " + str(j)))
        model.setObjective(U, g.GRB.MAXIMIZE)
        for i in range(M):
            model.addConstr(g.quicksum(
                [game_matrix[i, j] * sensor_path_vars[j] for j in range(N)]
            ) >= U, "cons1")
        model.addConstr(g.quicksum(sensor_path_vars) == 1, "cons2")
        for pv in sensor_path_vars:
            model.addConstr(pv >= 0, "cons3")
        model.optimize()
        sensor_distribution[0] = [var.X for var in sensor_path_vars]

        # BR
        sensor_br, _ = best_sensor(gallery, path_distribution)
        sensor_val = sensor_value(gallery, sensor_br, path_distribution)
        # BR
        tmp_C = cost_function(gallery, sensor_distribution)
        path_br, _ = best_plan(gallery, tmp_C)
        planner_val = planner_value(gallery, path_br, sensor_distribution)

        delta = abs(planner_val - sensor_val)
        if delta > epsilon:
            if path_br in path_distribution[1]:
                pass
            else:
                path_distribution[0].append(0.0)
                path_distribution[1].append(path_br)
                # Lengthen row dimension by 1.
                game_matrix = np.pad(game_matrix, ((0, 1), (0, 0)))
                for j in range(game_matrix.shape[1]):
                    sensor_idx = sensor_distribution[1][j]
                    game_matrix[-1, j] = pair_evaluation(gallery, path_br, sensor_idx)

            if sensor_br in sensor_distribution[1]:
                pass
            else:
                sensor_distribution[0].append(0.0)
                sensor_distribution[1].append(sensor_br)
                # Lengthen column dimension by 1.
                game_matrix = np.pad(game_matrix, ((0, 0), (0, 1)))
                for i, path in enumerate(path_distribution[1]):
                    game_matrix[i, -1] = pair_evaluation(gallery, path, sensor_br)
        else:
            break
    return sensor_distribution[0], sensor_distribution[1], path_distribution[0], path_distribution[1]


def planner_value(gallery, path, sensor_distribution):
    cost = 0
    for sensor_prob, sensor in zip(sensor_distribution[0], sensor_distribution[1]):
        cost += sensor_prob * pair_evaluation(gallery, path, sensor)
    return cost


def sensor_value(gallery, sensor, planner_distribution):
    cost = 0
    for path_prob, path in zip(planner_distribution[0], planner_distribution[1]):
        cost += path_prob * pair_evaluation(gallery, path, sensor)
    return cost
