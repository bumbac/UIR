#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time
import numpy as np
import threading as thread

#hexapod robot 
import hexapod_sim.HexapodHAL as hexapodhal

#cpg network
import cpg.oscilator_network as osc

#import robot parameters
from HexapodRobotConst import *

#import controller
import HexapodController as cntrl

#import messages
from messages import *

class HexapodRobot:
    def __init__(self, robot_id):
        """Hexapod controller class constructor
        """
        #robotHAL instance
        self.robot = hexapodhal.HexapodHAL(robot_id, TIME_FRAME)
        #controller instance
        self.controller = cntrl.HexapodController()
        self.control_method = self.controller.goto

        #gait parametrization
        self.v_left = 0
        self.v_right = 0
        
        #stride_length
        self.stride_length = 1
        #stride_height
        self.stride_height = 1

        #locomotion and navigation variables
        self.locomotion_stop = False
        self.locomotion_lock = thread.Lock()     #mutex for access to turn commands
        self.locomotion_status = False

        self.navigation_stop = False
        self.navigation_lock = thread.Lock()   #mutex for access to navigation coordinates
        self.navigation_status = False 
        self.navigation_goal = None

        #simulator data
        self.odometry_ = None
        self.collision_ = None
        self.laser_scan_ = None

    ########################################################################

    def turn_on(self):
        """Method to drive the robot into the default position
        """
        #read out the current pose of the robot
        configuration = self.robot.get_all_servo_position()

        #interpolate to the default position
        INTERPOLATION_TIME = 3000 #ms
        interpolation_steps = int(INTERPOLATION_TIME/TIME_FRAME)

        speed = np.zeros(18)
        for i in range(0,18):
            speed[i] = (SERVOS_BASE[i]-configuration[i])/interpolation_steps
        
        #execute the motion
        for t in range(0, interpolation_steps):
            self.robot.set_all_servo_position(configuration + t*speed)


    def turn_off(self):
        """Method to turn off the robot
        """
        self.robot.stop_simulation()

    ########################################################################
    ## LOCOMOTION
    ########################################################################

    def start_locomotion(self):
        """Method to start the robot locomotion 
        """
        #starting the locomotion thread if it is not running
        if self.locomotion_status == False: 
            print("Starting locomotion thread")
            try:
                self.locomotion_stop = False
                locomotion_thread = thread.Thread(target=self.locomotion)
            except:
                print("Error: unable to start locomotion thread")
                sys.exit(1)

            locomotion_thread.start()
        else:
            print("The locomotion is already running")

    def stop_locomotion(self):
        """Method to stop the locomotion
        """
        print("Stopping the locomotion thread")
        self.locomotion_stop = True

    def locomotion(self):
        """Method for locomotion control of the hexapod robot
        """
        self.locomotion_status = True
        
        #cpg network instantiation
        cpg = osc.OscilatorNetwork(6)
        cpg.change_gait(osc.TRIPOD_GAIT_WEIGHTS)

        coxa_angles= [0, 0, 0, 0, 0, 0]
        cycle_length = [0, 0, 0, 0, 0, 0]
       
        last_timestamp = 0
        #main locomotion control loop
        while not self.locomotion_stop:
            # steering - acquire left and right steering speeds
            self.locomotion_lock.acquire()
            left = np.min([1, np.max([-1, self.v_left])])
            right = np.min([1, np.max([-1, self.v_right])])
            self.locomotion_lock.release()

            coxa_dir = [left, left, left, right, right, right]      #set directions for individual legs

            #next step of CPG
            cycle = cpg.oscilate_all_CPGs()
            #read the state of the network
            data = cpg.get_last_values()

            #reset coxa angles if new cycle is detected
            for i in range(0, 6):
                cycle_length[i] += 1;
                if cycle[i] == True:
                    coxa_angles[i]= -((cycle_length[i]-2)/2)*COXA_MAX
                    cycle_length[i] = 0

            angles = np.zeros(18)
            #calculate individual joint angles for each of six legs
            for i in range(0, 6):
                femur_val = FEMUR_MAX*data[i] #calculation of femur angle
                if femur_val < 0:
                    coxa_angles[i] -= coxa_dir[i]*COXA_MAX  #calculation of coxa angle -> stride phase
                    femur_val *= 0.025
                else:
                    coxa_angles[i] += coxa_dir[i]*COXA_MAX  #calculation of coxa angle -> stance phase
                
                coxa_val = coxa_angles[i]
                tibia_val = -TIBIA_MAX*data[i]       #calculation of tibia angle
                    
                #set position of each servo
                angles[COXA_SERVOS[i] - 1] = SIGN_SERVOS[i]*coxa_val*self.stride_length + COXA_OFFSETS[i]
                angles[FEMUR_SERVOS[i] - 1] = SIGN_SERVOS[i]*femur_val*self.stride_height + FEMUR_OFFSETS[i]
                angles[TIBIA_SERVOS[i] - 1] = SIGN_SERVOS[i]*tibia_val*self.stride_height + TIBIA_OFFSETS[i]
                    
            #set all servos simultaneously
            self.robot.set_all_servo_position(angles)

            # GET DATA FROM SIMULATOR
            self.odometry_ = self.robot.get_robot_odometry()
            self.collision_ = self.robot.get_robot_collision()
            self.laser_scan_ = self.robot.get_laser_scan()

            time.sleep(TIME_FRAME/100.0)
            
        self.locomotion_status = False    
    
    def move(self, cmd):
        """Function to set diferential steering command
        Args:
            cmd: Twist velocity command
        """
        if cmd == None:
            left = 0
            right = 0
            self.navigation_goal = None
        else:
            #translate the velocity command into the differential steering command
            linear_x = np.min([np.max([cmd.linear.x,-1]),1])
            linear_y = np.min([np.max([cmd.linear.y,-1]),1])
            angular_z = np.min([np.max([cmd.angular.z,-1]),1])
            
            right = (linear_x + angular_z)/2
            left = linear_x - right

        self.locomotion_lock.acquire()
        self.v_left = SPEEDUP*left
        self.v_right = SPEEDUP*right
        self.locomotion_lock.release()

    ########################################################################
    ## NAVIGATION
    ########################################################################

    def start_navigation(self):
        """Method to start the navigation thread, that will guide the robot towards the goal set using the goto function
        """
        #starting the navigation thread if it is not running
        if self.navigation_status == False: 
            print("Starting navigation thread")
            try:
                self.navigation_stop = False
                navigation_thread = thread.Thread(target=self.navigation)
            except:
                print("Error: unable to start navigation thread")
                sys.exit(1)

            navigation_thread.start()
        else:
            print("The navigation is already running")
        
    def stop_navigation(self):
        """
        Stop the navigation thread
        """
        print("Stopping the navigation thread")
        self.navigation_stop = True
        self.locomotion_stop = True

    def navigation(self):
        """Method for the navigation control towards a given goal
        """
        self.navigation_status = True

        #start the locomotion if it is not running
        if self.locomotion_status == False:
            print("locomotion thread is not yet running -- starting it")
            self.start_locomotion()

        #navigation loop
        cmd = None
        while not self.navigation_stop:
            #open-loop control
            if self.control_method == self.controller.goto:
                cmd = self.control_method(
                            goal = self.navigation_goal,
                            odometry = self.odometry_,
                            collision = self.collision_
                      )
            #reactive control
            elif self.control_method == self.controller.goto_reactive:
                cmd = self.control_method(
                            goal = self.navigation_goal,
                            odometry = self.odometry_,
                            collision = self.collision_,
                            laser_scan = self.laser_scan_
                      )

            #drive the robot using the selected command
            self.move(cmd)
            #pause for a bit 
            time.sleep(0.1)

        self.navigation_status = False
            
    def goto(self, goal):
        """open-loop navigation towards a selected navigational goal
        Args:
            goal: Pose of the robot goal 
        """
        self.control_method = self.controller.goto
        self.navigation_lock.acquire()
        self.navigation_goal = goal
        self.navigation_lock.release()

    def goto_reactive(self, goal):
        """navigation towards a selected navigational goal while avoiding obstacles
        Args:
            goal: Pose of the robot goal 
        """
        self.control_method = self.controller.goto_reactive
        self.navigation_lock.acquire()
        self.navigation_goal = goal
        self.navigation_lock.release()

if __name__=="__main__":
    robot = HexapodController()
    #drive the robot into the default position
    robot.turn_on()
    time.sleep(3)

    #test the locomotion
    robot.start_locomotion()
    time.sleep(3)
    
    #velocity command
    #go straight
    cmd = Twist()
    cmd.linear.x = 1.0

    robot.move(cmd)
    time.sleep(10)

    #turn
    cmd.linear.x = 0.0
    cmd.angular.z = 1.0
    robot.move(cmd)
    time.sleep(10)
    
    robot.stop_locomotion()
    robot.turn_off()
