#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import pathlib as pth
import platform
import sys
import os
import ctypes as ct

import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv

#import communication messages
from messages import *

class Environment:
    class Mesh:
        def __init__(self):
            self.vertices = []
            self.faces = []
            self.limits_x = (0,0)
            self.limits_y = (0,0)
            self.limits_z = (0,0)

        def load_mesh(self, filename):
            """ 
            Load mesh from file 
            
            Parameters
            ----------
            filename : str 
                path to the mesh object file
            """
            with open(str(filename),'r') as f:
                for line in f:
                    q = line.split(' ')
                    if q[0] == "v":
                        #add vertex
                        self.vertices.append([float(q[1]),float(q[2]),float(q[3])])
                    elif q[0] == "f":
                        #add face
                        e1 = q[1].split('//')[0]
                        e2 = q[2].split('//')[0]
                        e3 = q[3].split('//')[0]
                        self.faces.append([int(e1)-1,int(e2)-1,int(e3)-1])
                    elif q[0] == "dx":
                        #x dimension of the map
                        self.limits_x = (float(q[1]),float(q[2]))
                    elif q[0] == "dy":
                        #y dimension of the map
                        self.limits_y = (float(q[1]),float(q[2]))
                    elif q[0] == "dz":
                        #z dimension of the map
                        self.limits_z = (float(q[1]),float(q[2]))
                    else:
                        #do nothing and continue
                        continue
        
    def __init__(self):
        self.obstacle = self.Mesh()
        self.robot = self.Mesh()

        #initialize the RAPID collision checking library
        self.librapid = None

        try:
            file_extension = '.so'
            if platform.system() =='cli':
              file_extension = '.dll'
            elif platform.system() =='Windows':
                file_extension = '.dll'
            elif platform.system() == 'Darwin':
                file_extension = '.dylib'
            else:
                file_extension = '.so'

            libfullpath = os.path.abspath(os.path.join(os.path.dirname(__file__), 'libRAPID' + file_extension))
            print("loading " + libfullpath)
            self.librapid = ct.CDLL(libfullpath)
        except:
            print ('----------------------------------------------------')
            print ('The rapid collision checking library could not be loaded.')
            print ('----------------------------------------------------')
        
        self.librapid.new_RapidAPI.restype = ct.c_void_p
        self.rapid_handle = self.librapid.new_RapidAPI()

        #functions api
        self.rapid_add_obstacle_face = ct.CFUNCTYPE(None, ct.c_void_p, ct.c_double*3, ct.c_double*3, ct.c_double*3)(("add_obstacle_face", self.librapid))
        self.rapid_add_robot_face = ct.CFUNCTYPE(None, ct.c_void_p, ct.c_double*3, ct.c_double*3, ct.c_double*3)(("add_robot_face", self.librapid))
        self.rapid_finish_faces = ct.CFUNCTYPE(None, ct.c_void_p)(("finish_faces", self.librapid))
        self.rapid_check_collision = ct.CFUNCTYPE(ct.c_bool, ct.c_void_p, ct.c_double*16)(("check_collision", self.librapid))


    def load_environment(self, problem_name):
        """
        Load the obstacles and the robot given the fproblem name

        Parameters
        ----------
        problem_name: String
            name of the problem
        """
        obstacle_file = pth.Path(problem_name + "_obstacle.obj")
        robot_file = pth.Path(problem_name + "_robot.obj")
        if obstacle_file.is_file() and robot_file.is_file():
            #if both files exists load them
            self.obstacle.load_mesh(obstacle_file)
            self.robot.load_mesh(robot_file)

        #transfer the models into the RAPID
        for face in self.obstacle.faces:
            p1 = self.obstacle.vertices[face[0]]
            p2 = self.obstacle.vertices[face[1]]
            p3 = self.obstacle.vertices[face[2]]

            self.add_face_collision_check("obstacle", p1, p2, p3)

        for face in self.robot.faces:
            p1 = self.robot.vertices[face[0]]
            p2 = self.robot.vertices[face[1]]
            p3 = self.robot.vertices[face[2]]

            self.add_face_collision_check("robot", p1, p2, p3)
        
        #finish the models for collision checking
        self.rapid_finish_faces(self.rapid_handle)
        
            
    def add_face_collision_check(self, model, p1, p2, p3):
        """
        helper function to load models into the collision checker

        Parameters
        ----------
        model: String
            type of the model being loaded - "obstacle" or "robot"
        p1: numpy array(float*3)
        p2: numpy array(float*3)
        p3: numpy array(float*3)
            coordinates of the faces
        """
        arrtype = ct.c_double * 3
        arr_p1 = arrtype()
        arr_p2 = arrtype()
        arr_p3 = arrtype()
        for i in range(0,3):
            arr_p1[i] = p1[i]
            arr_p2[i] = p2[i]
            arr_p3[i] = p3[i]
        if model == "obstacle":
            self.rapid_add_obstacle_face(self.rapid_handle, arr_p1, arr_p2, arr_p3)
        elif model == "robot":
            self.rapid_add_robot_face(self.rapid_handle, arr_p1, arr_p2, arr_p3)


    def check_robot_collision(self, T):
        """
        Check the robot collision with the obstacles given the transformation of the robot pose Pose 

        Parameters
        ----------
        T: Pose - robot pose in SE(3)

        Returns
        -------
        bool
            True if there is collision, False otherwise
        """
        arrtype = ct.c_double * 16
        arr_P = arrtype()
        P = np.reshape(T.to_T(),(16,1))
        for i in range(0,16):
            arr_P[i] = P[i]
        #check for the collision
        ret = self.rapid_check_collision(self.rapid_handle, arr_P)
        return ret

    def set_limits(self, limits):
        self.limits_x = limits[0]
        self.limits_y = limits[1]
        self.limits_z = limits[2]

    ##############################################
    ## Helper functions to plot the environment
    ##############################################
    def plot_environment(self, ax, P):
        """Method to plot the environment
        Args:  ax: axes object
               P: Pose - Robot pose in SE(3)
        Returns: ax axes object
        """
        #project the robot to the right position
        T = P.to_T()
        R = T[0:3,0:3]
        t = T[0:3,3]
        vr = [np.matmul(R,np.asarray(x)) + t for x in self.robot.vertices]

        #plot obstacles and the robot
        # this plotting is necessary to have proper depth visualization 
        xx = np.asarray([x[0] for x in self.obstacle.vertices] + [x[0] for x in vr])
        yy = np.asarray([x[1] for x in self.obstacle.vertices] + [x[1] for x in vr])
        zz = np.asarray([x[2] for x in self.obstacle.vertices] + [x[2] for x in vr])
 
        #offset the robot faces in the plot for the number of obstacle vertices
        idx = len(self.obstacle.vertices)
        fr = [np.asarray(x)+np.array([idx,idx,idx]) for x in self.robot.faces]
        fc = self.obstacle.faces + fr

        if ax.name=="3d":
            #3D plot the environment
            ax.plot_trisurf(xx,yy,zz, triangles=fc, color="blue", shade=True, edgecolor= None, alpha=0.5)

        else:
            #2D plot
            for face in fc:
                pp = plt.Polygon([[xx[face[0]],yy[face[0]]],
                                 [xx[face[1]],yy[face[1]]],
                                 [xx[face[2]],yy[face[2]]]])
                ax.add_patch(pp)

        self.plot_limits(ax)

        return ax

    def plot_limits(self, ax):
        points = []
        if self.limits_z[0] == self.limits_z[1]:
            for x in self.limits_x:
                for y in self.limits_y:
                    points.append(np.array([x,y]))

            for p1 in points:
                for p2 in points:
                    if np.sum((p1 - p2) == 0) == 1:
                        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], "-k", linewidth=1)

        else:
            for x in self.limits_x:
                for y in self.limits_y:
                    for z in self.limits_z:
                        points.append(np.array([x,y,z]))

            for p1 in points:
                for p2 in points:
                    if np.sum((p1 - p2) == 0) == 2:
                        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], "-k", linewidth=1)

