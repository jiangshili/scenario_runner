import numpy as np
import scipy
import sys
import os
import glob
import torch
import math

from scipy.misc import imresize

import matplotlib.pyplot as plt

from srunner.challenge.agents.network import CoILModel

try:
    sys.path.append(glob.glob('**/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

from srunner.challenge.agents.autonomous_agent import AutonomousAgent

from enum import Enum


class RoadOption(Enum):
    """
    RoadOption represents the possible topological configurations when moving from a segment of lane to other.
    """
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4



def distance_vehicle(waypoint, vehicle_transform):
    loc = vehicle_transform.location
    dx = waypoint['lat'] - loc.x
    dy = waypoint['lon'] - loc.y

    return math.sqrt(dx * dx + dy * dy)

class CoILAgent(AutonomousAgent):

    def __init__(self):
        AutonomousAgent.__init__(self)
        self._params = {
            'checkpoint': '200000',
            "model_type": 'coil-icra',
            "model_configuration": {'perception': {
                'res': {
                    'name': 'resnet34',
                    'num_classes': 512
                }
            },
                'measurements': {
                    'fc': {
                        'neurons': [128, 128],
                        'dropouts': [0.0, 0.0]
                    }
                },
                'join': {
                    'fc': {
                        'neurons': [512],
                        'dropouts': [0.0]
                    }
                },
                'speed_branch': {
                    'fc': {
                        'neurons': [256, 256],
                        'dropouts': [0.0, 0.5]
                    }
                },
                'branches': {
                    'number_of_branches': 4,
                    'fc': {
                        'neurons': [256, 256],
                        'dropouts': [0.0, 0.5]
                    }
                }
            },
            'image_cut': [90, 485],
            'speed_factor': 12.0,
            'size': [3, 88, 200]
        }

        self._checkpoint = torch.load(os.path.join('srunner/challenge/agents/network',
                                                   str(self._params['checkpoint']) + '.pth'))
        # Set the carla version that is going to be used by the interface
        # We save the checkpoint for some interesting future use.

        self._model = CoILModel(self._params["model_type"],
                                self._params["model_configuration"])
        self.first_iter = True
        # Load the model and prepare set it for evaluation
        self._model.load_state_dict(self._checkpoint['state_dict'])
        self._model.cuda()
        self._model.eval()

        self.latest_image = None
        self.latest_image_tensor = None

        # Number of expanded curve commands, both in front and in back

        self._expand_command_front = 5
        self._expand_command_back = 3

    def sensors_setup(self):

        sensors = [['sensor.camera.rgb',
                   {'x': 2.0, 'y': 0.0,
                    'z': 1.40, 'roll': 0.0,
                    'pitch': -15.0, 'yaw': 0.0,
                    'width': 800, 'height': 600,
                    'fov': 100},
                    'rgb'],
                   #['sensor.speedometer',
                   # {'reading_frequency': 20},
                   # 'speed'
                   # ],
                   ['sensor.other.gnss', {'x': 0.7, 'y': -0.4, 'z': 1.60},
                    'GPS']
                   ]

        return sensors

    # CONVERT FROM LAT LON TO WAYPOINTS IN Z.

    def run_step(self, input_data):

        # TODO, input speed should come on the input data

        #measurements, sensor_data, directions, target
        print ("Input data SPEED")
        #print (input_data['speed'])

        directions = self._get_current_direction(input_data['GPS'])

        # Take the forward speed and normalize it for it to go from 0-1
        #norm_speed = input_data['speed'][1] / self._params['speed_factor'] #.SPEED_FACTOR
        norm_speed = 0.3
        norm_speed = torch.cuda.FloatTensor([norm_speed]).unsqueeze(0)
        directions_tensor = torch.cuda.LongTensor([directions])
        # Compute the forward pass processing the sensors got from CARLA.
        model_outputs = self._model.forward_branch(self._process_sensors(input_data['rgb'][1]), norm_speed,
                                                   directions_tensor)

        steer, throttle, brake = self._process_model_outputs(model_outputs[0])

        control = carla.VehicleControl()

        control.steer = float(steer)
        control.throttle = float(throttle)
        control.brake = float(brake)
        # There is the posibility to replace some of the predictions with oracle predictions.

        self.first_iter = False

        return control

    def set_global_plan(self, topological_plan):
        # We expand the commands before the curves.

        self._expand_commands(topological_plan)

        self._global_plan = topological_plan


    def get_attentions(self, layers=None):
        """

        Returns
            The activations obtained from the first layers of the latest iteration.

        """
        if layers is None:
            layers = [0, 1, 2]
        if self.latest_image_tensor is None:
            raise ValueError('No step was ran yet. '
                             'No image to compute the activations, Try Running ')
        all_layers = self._model.get_perception_layers(self.latest_image_tensor)
        cmap = plt.get_cmap('inferno')
        attentions = []
        for layer in layers:
            y = all_layers[layer]
            att = torch.abs(y).mean(1)[0].data.cpu().numpy()
            att = att / att.max()
            att = cmap(att)
            att = np.delete(att, 3, 2)
            attentions.append(imresize(att, [88, 200]))
        return attentions

    def _process_sensors(self, sensor):

        iteration = 0

        sensor = sensor[self._params['image_cut'][0]:self._params['image_cut'][1], ...]

        sensor = scipy.misc.imresize(sensor, (self._params['size'][1], self._params['size'][2]))

        self.latest_image = sensor

        sensor = np.swapaxes(sensor, 0, 1)

        sensor = np.transpose(sensor, (2, 1, 0))

        sensor = torch.from_numpy(sensor / 255.0).type(torch.FloatTensor).cuda()

        if iteration == 0:
            image_input = sensor

        iteration += 1

        image_input = image_input.unsqueeze(0)

        self.latest_image_tensor = image_input

        return image_input

    def _get_current_direction(self, vehicle_transform):

        # TODO: probably start by expanding the size of the turns.

        # for the current position and orientation try to get the closest one from the waypoints
        closest_id = 0
        min_distance = 100000
        for index in self._global_plan:

            waypoint = self._global_plan[index][0]
            # TODO maybe add if the agent is in a similar orientation.

            computed_distance = distance_vehicle(waypoint, vehicle_transform)
            if computed_distance < min_distance:
                min_distance = computed_distance
                closest_id = index

        direction = self._global_plan[closest_id][1]

        if direction == RoadOption.LEFT:
            direction = 3.0
        elif direction == RoadOption.RIGHT:
            direction = 4.0
        elif direction == RoadOption.STRAIGHT:
            direction = 5.0
        else:
            direction = 2.0

        return direction

    def _process_model_outputs(self, outputs):
        """
         A bit of heuristics in the control, to eventually make car faster, for instance.
        Returns:

        """
        steer, throttle, brake = outputs[0], outputs[1], outputs[2]
        if brake < 0.05:
            brake = 0.0

        if throttle > brake:
            brake = 0.0


        return steer, throttle, brake

    def _expand_commands(self, topological_plan):
        """ The idea is to make the intersection indications to last longer"""

        # O(2*N) algorithm , probably it is possible to do in O(N) with queues.

        # Get the index where curves start and end
        curves_start_end = []
        inside = False
        start = -1
        current_curve = RoadOption.LANEFOLLOW
        for index in range(len(topological_plan)):

            command = topological_plan[index][1]

            if command != RoadOption.LANEFOLLOW and not inside:
                inside = True
                start = index
                current_curve = command

            if command == RoadOption.LANEFOLLOW and inside:
                inside = False
                # End now is the index.
                curves_start_end.append([start, index, current_curve])
                if start == -1:
                    raise ValueError("End of curve without start")

                start = -1

        for start_end_index_command in curves_start_end:
            start_index = start_end_index_command[0]
            end_index = start_end_index_command[0]
            command  = start_end_index_command[0]

            # Add the backwards curves ( Before the begginning)
            for index in range(1, self._expand_command_front+1):
                changed_index = start_index - index
                if changed_index > 0:
                    topological_plan[changed_index] = (topological_plan[changed_index][0], command)


            for index in range(0, self._expand_command_back):
                changed_index = end_index + index
                if changed_index < len(topological_plan) :
                    topological_plan[changed_index][1] = (topological_plan[changed_index][0], command)


