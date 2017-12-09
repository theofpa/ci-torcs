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

        # A vector to fill with sensor data
        self.sensors = np.zeros(22)

        # An echo state network to transform sensor data
        self.esn = pickle.load(open("reservoir.p", "rb"))

        # A neural classifier of steering actions based on echos from the esn
        self.steer_model = FFNNClassifier(200, 250, 7)
        self.steer_model.load_state_dict(load("steer_model_v1.pt"))

        # A self organizing map to determine the state of the car
        self.som = pickle.load(open("som_8x8.p", "rb"))

        # The speeds to set based on the state of the car
        self.map_speeds = np.load("map_speeds_8x8.npy")

    def drive(self, carstate: State) -> Command:
        command = Command()

        # Fill sensor data
        self.sensors[0] = carstate.speed_x
        self.sensors[1] = carstate.distance_from_center
        self.sensors[2] = carstate.angle * np.pi/180
        self.sensors[3:] = np.array(carstate.distances_from_edge)

        # Input to steering reservoir
        echo = self.esn.transform(self.sensors.reshape(1, 22))
        if abs(carstate.distance_from_center) < 1:
            # Readout reservoir using neural network, predict and set steering
            steer_idx = self.steer_model.predict(echo)
            command.steering = MyDriver.steering_levels[steer_idx]

            # Determine car state using SOM to set the speed
            car_state = self.som.get_closer(self.sensors[3:, np.newaxis])
            target_speed = self.map_speeds[car_state]
            self.accelerate(carstate, target_speed, command)
        else:
            self.steer(carstate, 0, command)
            self.accelerate(carstate, 20, command)

        return command
