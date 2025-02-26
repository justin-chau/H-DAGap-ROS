
from builtins import object
from matrix import matrix
import random
import numpy as np
import math
import pdb

state_num = 6
init_covariance = 1000
dt = 0.1
# external motion
u = matrix([[0.], [0.], [0.], [0.], [0.], [0.]])
# next state function
F =  matrix([[1., dt, 0.5*( dt**2 ), 0., 0., 0.], [0., 1., dt, 0., 0., 0.], [0., 0., 1., 0., 0., 0.],
             [0., 0., 0., 1., dt, 0.5*( dt**2 )], [0., 0., 0., 0., 1., dt], [0., 0., 0., 0., 0., 1.]])
single_integrator_F =  matrix([[1., dt, 0, 0., 0., 0.], [0., 1., dt, 0., 0., 0.], [0., 0., 1., 0., 0., 0.],
             [0., 0., 0., 1., dt, 0], [0., 0., 0., 0., 1., dt], [0., 0., 0., 0., 0., 1.]])
# measurement function: reflect the fact that we observe x and y 
H =  matrix([[1., 0., 0., 0., 0., 0.], [0., 0., 0., 1., 0., 0.]])
# measurement uncertainty: use 6x6 matrix with 0 as main diagonal
R =  matrix([[0.01, 0.], [0., 0.01]])  #matrix([[0.1, 0.], [0., 0.1]]) 
# 6d identity matrix
I =  matrix(np.eye(state_num)) 

class KalmanFilter(object):

    class ObstacleMeasure(object):        
        def __init__(self):
            # x, x', x'', y, y', y''
            self.estimate = matrix([np.array([0] * state_num)]).transpose()
            # init covariance between x, x', x'', y, y', y''
            self.covariance = matrix(np.eye(state_num) * init_covariance)

    def __init__(self):
        self.obstacle_locations = []
        self.obstacle_measure = {}
        
    def observe_obstacles(self, obstacle_locations):
        """ self - pointer to the current object.
           asteroid_locations - a list of asteroid observations. Each 
           observation is a tuple (i,x,y) where i is the unique ID for
           an asteroid, and x,y are the x,y locations (with noise) of the
           current observation of that asteroid at this timestep.
           Only asteroids that are currently 'in-bounds' will appear
           in this list, so be sure to use the asteroid ID, and not
           the position/index within the list to identify specific
           asteroids. (The list may change in size as asteroids move
           out-of-bounds or new asteroids appear in-bounds.)

           Return Values:
                    None
        """
        measure_locations = []
        for location in obstacle_locations:
            ID, x, y = location
            observation = matrix([[x], [y]])
            if (ID not in self.obstacle_measure):
                self.obstacle_measure[ID] = self.ObstacleMeasure()
            estimate = self.obstacle_measure[ID].estimate
            covariance = self.obstacle_measure[ID].covariance
            # measurement update
            y = observation - (H * estimate)            
            S = H * (covariance) * H.transpose() + R
            K = covariance * H.transpose() * S.inverse()
            #pdb.set_trace()
            estimate = estimate + (K * y)
            #print(f"covariance {covariance}")
            #pdb.set_trace()
            covariance = (I - (K * H)) * covariance
            
            # add asteroids 
            measure_locations.append((ID, estimate[0][0], estimate[3][0]))
            self.obstacle_measure[ID].estimate = estimate                
            self.obstacle_measure[ID].covariance = covariance
        
        self.obstacle_locations = measure_locations

    def distance(self, robot_x, robot_y, obs_x, obs_y):
        """        
        return: float value - means the distance between craft and asteroid
        """
        diff_x = robot_x - obs_x
        diff_y = robot_y - obs_y
        return math.sqrt(diff_x ** 2 + diff_y ** 2)
    
    def estimate_obstacle_locs(self):
        """ Should return an iterable (list or tuple for example) that
            contains data in the format (i,x,y), consisting of estimates
            for all in-bound asteroids. """
        estimate_locations = []
        for location in self.obstacle_locations:
            ID, x, y = location
            if (ID not in self.obstacle_measure):
                self.obstacle_measure[ID] = self.ObstacleMeasure()
            estimate = self.obstacle_measure[ID].estimate
            covariance = self.obstacle_measure[ID].covariance
            # prediction
            estimate = (F * estimate)
            covariance = F * covariance * F.transpose()
            
            estimate_locations.append((ID, estimate[0][0], estimate[3][0]))
            self.obstacle_measure[ID].estimate = estimate
            self.obstacle_measure[ID].covariance = covariance     

        return estimate_locations
