from pytocl.driver import Driver
from pytocl.car import State, Command

import pickle
import numpy as np
from simple_esn import SimpleESN
from ffnn_classifier import FFNNClassifier
from torch import load
from som import SelfOrganizingMap

class MyDriver(Driver):
    steering_levels = [-1, -3/4, -1/3, 0, 1/3, 3/4, 1]

    def __init__(self, logdata=True):
        super(MyDriver, self).__init__(logdata)

        self.esn = pickle.load(open("reservoir.p", "rb"))
        self.steer_model = FFNNClassifier(200, 250, 7)
        self.steer_model.load_state_dict(load("steer_model_v1.pt"))
        self.som = pickle.load(open("som.p", "rb"))
        self.sensors = np.zeros(22)

    def drive(self, carstate: State) -> Command:
        command = Command()
        
        # Fill input data
        self.sensors[0] = carstate.speed_x
        self.sensors[1] = carstate.distance_from_center
        self.sensors[2] = carstate.angle * np.pi/180
        self.sensors[3:] = np.array(carstate.distances_from_edge)
        
        # Input to steering reservoir
        echo = self.esn.transform(self.sensors.reshape(1, 22))
        
        # Readout reservoir using neural network and set steering       
        steer = MyDriver.steering_levels[self.steer_model.predict(echo)]        
        command.steering = steer

        # Determine car state using SOM to set the speed
        edge_state = self.som.get_closer(self.sensors[3:, np.newaxis])        
        if edge_state >= 15 and edge_state <= 24:
            v_x = 230
        elif edge_state in [0, 2, 7, 18, 7, 6, 9, 8]:            
            v_x = 100
        elif edge_state in [3, 4]:
            v_x = 50
        else:
            v_x = 150
        self.accelerate(carstate, v_x, command)

        print(carstate.opponents)

        return command