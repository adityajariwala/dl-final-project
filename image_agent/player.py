import numpy as np
import torch
from PIL import Image
from os import path
from torchvision import transforms
import random
import sys

# Import model objects here as well once created

if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

GOALS = np.float32([[0, 75], [0, -75]])

LOST_STATUS_STEPS = 10
LOST_COOLDOWN_STEPS = 10
START_STEPS = 25
LAST_PUCK_DURATION = 4
MIN_SCORE = 0.2
MAX_DET = 15
MAX_DEV = 0.7
MIN_ANGLE = 20
MAX_ANGLE = 120
TARGET_SPEED = 15
STEER_YIELD = 15
DRIFT_THRESH = 0.7
TURN_CONE = 100

TO_RAD = np.pi / 180


def screen_to_world_coordinates(puck_screen_coordinates, proj, view, normalization_factor):
    WH2 = np.array([400, 300]) / 2

    normalized_puck_screen_coordinates = puck_screen_coordinates
    normalized_puck_screen_coordinates[0] = normalized_puck_screen_coordinates[0] / WH2[0] - 1
    normalized_puck_screen_coordinates[1] = normalized_puck_screen_coordinates[1] / WH2[1] - 1

    puck_world_coordinates = np.linalg.inv(view.T) @ np.linalg.inv(proj) @ np.array(list(
        [normalization_factor * normalized_puck_screen_coordinates[0],
         -normalization_factor * normalized_puck_screen_coordinates[1]]) + [normalization_factor,
                                                                            1 - normalization_factor])
    return puck_world_coordinates[:3]


def norm(vector):
    return np.linalg.norm(vector)


class Team:
    agent_type = 'image'

    def __init__(self):

        # Initialize revelant logging parameters for karts
        self.kart_details = {}
        self.reverse_for_puck = {0: False, 1: False}
        self.reverse_for_puck_previous_frame = {0: False, 1: False}
        self.follow_through_after_reverse = {0: 0, 1: 0}
        self.puck_in_frame_previous_frame = {}
        self.puck_3d_coordinates_previous_frame = {}

        self.kart_loc_prev = {0: np.array([0,0,0]), 1: np.array([0,0,0])}

        self.total_frame = {0: 0, 1: 0}

        self.kart_image = {}

        self.too_close = {0: False, 1: False}

        self.forward_frames = {0: 0, 1: 0}

        self.puck_in_frame = {}
        self.puck_screen_coordinates = {}
        self.puck_3d_coordinates = {}

        self.kart_actions = {
            0: {
                'steer': 0,
                'acceleration': 1,
                'brake': False,
                'drift': False,
                'nitro': False,
                'rescue': False
            },
            1: {
                'steer': 0,
                'acceleration': 1,
                'brake': False,
                'drift': False,
                'nitro': False,
                'rescue': False
            }
        }

        # Initialization for the sake of testing model performance
        self.model_puck_classifier = torch.load(path.join(path.dirname(path.abspath(__file__)), 'model_puck_new_17.pt'),
                                                map_location="cpu").to(device)
        self.model_puck_classifier.eval()

        self.model_puck_unified = torch.load(path.join(path.dirname(path.abspath(__file__)), 'model_unified_new_25.pt'),
                                             map_location="cpu").to(device)
        self.model_puck_unified.eval()

        self.transform = transforms.Compose(
            [transforms.Resize([128, 128]), transforms.Grayscale(num_output_channels=3), transforms.ToTensor()])
        self.team = None
        self.num_players = None

    def new_match(self, team: int, num_players: int) -> list:
        """
        Let's start a new match. You're playing on a `team` with `num_players` and have the option of choosing your kart
        type (name) for each player.
        :param team: What team are you playing on RED=0 or BLUE=1
        :param num_players: How many players are there on your team
        :return: A list of kart names. Choose from 'adiumy', 'amanda', 'beastie', 'emule', 'gavroche', 'gnu', 'hexley',
                 'kiki', 'konqi', 'nolok', 'pidgin', 'puffy', 'sara_the_racer', 'sara_the_wizard', 'suzanne', 'tux',
                 'wilber', 'xue'. Default: 'tux'
        """
        """
           TODO: feel free to edit or delete any of the code below
        """
        self.team, self.num_players = team, num_players
        return ['tux'] * num_players

    def act(self, player_state, player_image):
        """
        This function is called once per timestep. You're given a list of player_states and images.

        DO NOT CALL any pystk functions here. It will crash your program on your grader.

        :param player_state: list[dict] describing the state of the players of this team. The state closely follows
                             the pystk.Player object <https://pystk.readthedocs.io/en/latest/state.html#pystk.Player>.
                             See HW5 for some inspiration on how to use the camera information.
                             camera:  Camera info for each player
                               - aspect:     Aspect ratio
                               - fov:        Field of view of the camera
                               - mode:       Most likely NORMAL (0)
                               - projection: float 4x4 projection matrix
                               - view:       float 4x4 view matrix
                             kart:  Information about the kart itself
                               - front:     float3 vector pointing to the front of the kart
                               - location:  float3 location of the kart
                               - rotation:  float4 (quaternion) describing the orientation of kart (use front instead)
                               - size:      float3 dimensions of the kart
                               - velocity:  float3 velocity of the kart in 3D

        :param player_image: list[np.array] showing the rendered image from the viewpoint of each kart. Use
                             player_state[i]['camera']['view'] and player_state[i]['camera']['projection'] to find out
                             from where the image was taken.

        :return: dict  The action to be taken as a dictionary. For example `dict(acceleration=1, steer=0.25)`.
                 acceleration: float 0..1
                 brake:        bool Brake will reverse if you do not accelerate (good for backing up)
                 drift:        bool (optional. unless you want to turn faster)
                 fire:         bool (optional. you can hit the puck with a projectile)
                 nitro:        bool (optional)
                 rescue:       bool (optional. no clue where you will end up though.)
                 steer:        float -1..1 steering angle
        """
        DistanceThreshold = 3
        ShootingAngleThreshold = 0.5  # (in steering terms)
        FollowThroughFrames = 30
        FramesDownTheLine = 2

        all_goals = np.array([[0, 75], [0, -75]])
        own_goal = all_goals[self.team - 1]
        opponents_goal = all_goals[self.team]
        own_corners = np.array([[50, all_goals[self.team - 1][1]], [-50, all_goals[self.team - 1][1]]])

        print_output = False

        # Loop through both Karts
        for i in [0, 1]:

            if print_output:
                print("Actions for Kart ", i)

            starting_frames = 50

            self.kart_details[i] = player_state[i]
            self.kart_image[i] = player_image[i]

            proj = self.kart_details[i]['camera']['projection']
            view = self.kart_details[i]['camera']['view']

            max_steer_angle = self.kart_details[i]['kart']['max_steer_angle']

            # loc_diff = abs(self.kart_loc_prev[i] - norm(self.kart_details[i]['kart']['front']))
            loc_diff = norm(self.kart_loc_prev[i] - np.array(self.kart_details[i]['kart']['front']))

            if loc_diff > 3 and self.total_frame[i] > starting_frames:
                if print_output:
                    print("GOAL SCORED, RESET! Diff: ", loc_diff)
                    print(loc_diff)
                self.total_frame = {0: 0, 1: 0}

            self.kart_loc_prev[i] = np.array(self.kart_details[i]['kart']['front'])
            velocity = norm(self.kart_details[i]['kart']['velocity'])
            rescue = velocity <= 0.01 or loc_diff <= 0.001

            fire_item = self.kart_details[i]['kart']['powerup']['num'] > 0

            # Model to Predict Puck's 3D Coordinates -
            prob_puck_in_frame = self.model_puck_classifier.forward(
                self.transform(Image.fromarray(self.kart_image[i]))[None, :, :, :].to(device))
            self.puck_in_frame[i] = torch.argmax(prob_puck_in_frame) == 1
            self.puck_in_frame[i] = self.puck_in_frame[i].cpu().detach().numpy().copy()

            if self.puck_in_frame[i]:
                puck_screen_coordinates, normalization_factor = self.model_puck_unified.forward(
                    self.transform(Image.fromarray(self.kart_image[i]))[None, :, :, :].to(device))
                puck_screen_coordinates = puck_screen_coordinates.cpu().detach().numpy()[0].copy()

                normalized_puck_screen_coordinates = puck_screen_coordinates.copy()
                normalized_puck_screen_coordinates[0] = normalized_puck_screen_coordinates[0] / 200 - 1
                normalized_puck_screen_coordinates[1] = normalized_puck_screen_coordinates[1] / 150 - 1

                # print("Predicted Screen Coordinates - ", puck_screen_coordinates, normalized_puck_screen_coordinates)

                normalization_factor = normalization_factor.cpu().detach().numpy()[0][0].copy()
                # print("Normalized Factor (Distance)", normalization_factor)

                puck_world_coordinates = screen_to_world_coordinates(puck_screen_coordinates, proj, view,
                                                                     normalization_factor)
                # print("Derived 3D Coordinates", puck_world_coordinates)
            else:
                puck_world_coordinates = np.array([0, 0, 0])
                normalized_puck_screen_coordinates = np.array([0, 0])

            self.puck_screen_coordinates[i] = np.array([0, 0])
            self.puck_3d_coordinates[i] = puck_world_coordinates.copy()
            normalized_puck_screen_coordinates = normalized_puck_screen_coordinates.copy()

            # Compute important directon vectors in world_cordinates & relevant distances -
            # 1. Kart direction vector (Kart's Orientation)
            kart_front = np.array(self.kart_details[i]['kart']['front'])[[0, 2]]
            kart_back = np.array(self.kart_details[i]['kart']['location'])[[0, 2]]
            kart_direction = (kart_front - kart_back) / np.linalg.norm(kart_front - kart_back)

            # 2. Puck direction vector & distance (wrt Kart's Back Position)
            puck_postion = np.array(self.puck_3d_coordinates[i])[[0, 2]]
            vector_kart_to_puck = np.array(puck_postion) - np.array(kart_back)
            kart_to_puck_direction = vector_kart_to_puck / np.linalg.norm(vector_kart_to_puck)
            distance_kart_to_puck = np.linalg.norm(vector_kart_to_puck)

            # 3. Opponent's Goal direction vector (wrt Puck's Position)
            # Deciding which of the two goals is ours depends on team initialization
            opponents_goal_position = all_goals[self.team]
            vector_puck_to_opponent_goal = np.array(opponents_goal_position) - np.array(puck_postion)
            puck_to_goal_direction = vector_puck_to_opponent_goal / np.linalg.norm(vector_puck_to_opponent_goal)

            # 4. Opponent's Goal direction vector (wrt Kart's Position)
            # Deciding which of the two goals is ours depends on team initialization
            vector_kart_to_opponent_goal = np.array(opponents_goal_position) - np.array(kart_back)
            kart_to_goal_direction = vector_kart_to_opponent_goal / np.linalg.norm(vector_kart_to_opponent_goal)
            distance_kart_to_opponent_goal = np.linalg.norm(vector_kart_to_opponent_goal)

            # 5. Kart's velocity vector and speed
            velocity_vector = np.array(self.kart_details[i]['kart']['velocity'])[[0, 2]]
            speed = np.linalg.norm(velocity_vector)

            # 6. Prediction of location of kart N frames down the line
            new_kart_back = kart_back + velocity_vector * FramesDownTheLine

            # 6a. New Kart Back to Puck location vector
            vector_new_kart_to_puck = np.array(puck_postion) - np.array(new_kart_back)
            new_kart_to_puck_direction = vector_new_kart_to_puck / np.linalg.norm(vector_new_kart_to_puck)

            # 6b. New Kart Back to Puck location vector
            vector_new_kart_to_opponent_goal = np.array(opponents_goal_position) - np.array(new_kart_back)
            new_kart_to_goal_direction = vector_new_kart_to_opponent_goal / np.linalg.norm(
                vector_new_kart_to_opponent_goal)

            # 7. We need to check if we're on the right side of field
            own_goal_position = all_goals[self.team - 1]
            vector_kart_to_own_goal = np.array(own_goal_position) - np.array(kart_back)
            kart_to_own_goal_direction = vector_kart_to_own_goal / np.linalg.norm(vector_kart_to_own_goal)
            distance_kart_to_own_goal = np.linalg.norm(vector_kart_to_own_goal)

            if np.abs(vector_kart_to_own_goal[1]) > 80:
                self.too_close[i] = False

            if print_output:
                print("Dist from own goal: ", vector_kart_to_own_goal)

            # --------------------------------------------------------------------------------------------------------
            # Actions start here
            # --------------------------------------------------------------------------------------------------------

            # If puck in frame, then we proceed further, otherwise we set actions to reverse, and steer in left
            if not self.puck_in_frame[i] and self.total_frame[i] > starting_frames:

                if print_output:
                    print('Puck Not in Frame - Go random direction..')

                # if print_output:
                #     print("Dist from own goal: ", distance_kart_to_own_goal)

                # threshold = 55 if self.too_close[i] else 30

                # Update here
                if not self.too_close[i] and np.abs(vector_kart_to_own_goal[1]) > 12:
                    angle_shooting_direction = np.degrees(np.arccos(np.dot(kart_direction, kart_to_own_goal_direction)))
                    if print_output:
                        print("Never mind, get closer to goal")
                    direction_of_steering = np.sign(np.cross(kart_direction, kart_to_own_goal_direction))
                    angle_shooting_direction = -direction_of_steering * angle_shooting_direction
                    steering_angle = angle_shooting_direction * np.pi / 180
                    steering_angle = np.clip(steering_angle, -1, 1)
                    brake = True
                    acceleration = 0.
                    self.forward_frames[i] = 0

                else:
                    self.too_close[i] = True
                    if self.forward_frames[i] < 20:
                        steering_angle = 0.8 if i == 1 else -0.8
                    elif self.forward_frames[i] < 50:
                        # steering_angle = 0
                        steering_angle = -0.6 if i == 1 else 0.6
                    else:
                        steering_angle = 0
                    self.forward_frames[i] += 1
                    brake = False
                    acceleration = 0.5

                self.kart_actions[i] = {
                    'steer': steering_angle,
                    'acceleration': acceleration,
                    'brake': brake,
                    'drift': False,
                    'fire': False,
                    'nitro': False,
                    'rescue': rescue
                }
                if print_output:
                    print("Final Action - ", self.kart_actions[i])
                    print()
                continue

            else:

                if print_output:
                    print('Puck in Frame. Setting Aim and Chasing')
                self.reverse_for_puck[i] = False

                # Reset puck's utilization basis partner's puck location
                # self.utilize_partners_puck_location[i] = False

                ### Implementation of Actions -
                angle_shooting_direction_goal = np.degrees(np.arccos(np.dot(kart_direction, kart_to_goal_direction))) # Both vectors have a norm 1, so dot represent cos(theta)
                direction_of_steering_goal = np.sign(np.cross(kart_direction, kart_to_goal_direction))
                # angle_shooting_direction_goal = -direction_of_steering_goal*angle_shooting_direction_goal

                if angle_shooting_direction_goal >= 120 or angle_shooting_direction_goal <= 20:

                  # To prevent glitching as we approach the puck
                  angle_shooting_direction = np.degrees(np.arccos(np.dot(kart_direction, kart_to_puck_direction))) # Both vectors have a norm 1, so dot represent cos(theta)
                  direction_of_steering = np.sign(np.cross(kart_direction, kart_to_puck_direction))
                  angle_shooting_direction = -direction_of_steering*angle_shooting_direction
                  steering_angle = (angle_shooting_direction)/90.
                  steering_angle = np.clip(steering_angle, -1, 1)

                  if np.abs(steering_angle) >= 0.1:

                    # The shooting direction is simple the angle between kart's orientation vector and vector from kart's back to puck
                    angle_shooting_direction = np.degrees(np.arccos(np.dot(kart_direction, new_kart_to_puck_direction))) # Both vectors have a norm 1, so dot represent cos(theta)
                    direction_of_steering = np.sign(np.cross(kart_direction, new_kart_to_puck_direction))
                    angle_shooting_direction = -direction_of_steering*angle_shooting_direction

                    if print_output:
                        print("Case 1")
                        print('Steering Angle - ',angle_shooting_direction)

                    steering_angle = (angle_shooting_direction)/90.
                    steering_angle = np.clip(steering_angle, -1, 1)

                else:
                  
                  ### Another Team's implementation ####
                  puck_y_coordinate = normalized_puck_screen_coordinates[1]
                  direction_of_steering_to_goal = np.sign(np.cross(kart_direction, kart_to_goal_direction))
                  angle_of_steering_to_goal = np.arccos(np.dot(kart_direction, kart_to_goal_direction))
                  angle_of_steering_to_goal = -direction_of_steering_to_goal*angle_of_steering_to_goal

                  norm_goal_distance = ((np.clip(distance_kart_to_opponent_goal, 10, 100) - 10) / 90) + 1
                  distance = 1 / norm_goal_distance ** 3
                  aim_point = puck_y_coordinate + np.sign(puck_y_coordinate - angle_of_steering_to_goal) * 0.3 * distance
                  steering_angle = np.clip(aim_point * 15, -1, 1)

                  # Then, we check if the goal and puck are on different sides of kart
                  # direction_of_steering_to_goal = np.sign(np.cross(kart_direction, kart_to_goal_direction))
                  # direction_of_steering_to_puck = np.sign(np.cross(kart_direction, new_kart_to_puck_direction))

                  # angle_of_steering_to_goal = np.arccos(np.dot(kart_direction, kart_to_goal_direction))
                  # angle_of_steering_to_puck = np.arccos(np.dot(kart_direction, new_kart_to_puck_direction))

                  # if direction_of_steering_to_goal*direction_of_steering_to_puck < 0:
                  #   # It indicates that movings towards goal might move away from puck, so first we get closer to puck till these angles align
                  #   angle_shooting_direction = np.degrees(np.arccos(np.dot(kart_direction, new_kart_to_puck_direction))) # Both vectors have a norm 1, so dot represent cos(theta)
                  #   direction_of_steering = np.sign(np.cross(kart_direction, new_kart_to_puck_direction))
                  #   angle_shooting_direction = -direction_of_steering*angle_shooting_direction

                  # else:
                  #   angle_shooting_direction_puck = np.degrees(np.arccos(np.dot(kart_direction, new_kart_to_puck_direction))) # Both vectors have a norm 1, so dot represent cos(theta)
                  #   direction_of_steering_puck = np.sign(np.cross(kart_direction, new_kart_to_puck_direction))
                  #   angle_shooting_direction_puck = -direction_of_steering_puck*angle_shooting_direction_puck

                  #   angle_shooting_direction_goal = np.degrees(np.arccos(np.dot(kart_direction, kart_to_goal_direction))) # Both vectors have a norm 1, so dot represent cos(theta)
                  #   direction_of_steering_goal = np.sign(np.cross(kart_direction, kart_to_goal_direction))
                  #   angle_shooting_direction_goal = -direction_of_steering_goal*angle_shooting_direction_goal

                  #   # weight = np.clip(distance_kart_to_opponent_goal, 1, 100)/100
                  #   # angle_shooting_direction = weight*angle_shooting_direction_puck + (1-weight)*angle_shooting_direction_goal

                  #   angle_shooting_direction = angle_shooting_direction_goal

                  # print("New Direction (Angle) -",angle_shooting_direction)
                  # # print("New Direction (Velocity) -",angle_shooting_direction_velocity)

                  # print("Case 2")
                  # print('Steering Angle - ',angle_shooting_direction)

                  # # Steering Angle - Treat these angles as 2X more steering
                  # steering_angle = 2*(angle_shooting_direction)/90.
                  # steering_angle = np.clip(steering_angle, -1, 1)

                ## b) Accelerate/Brake
                # Case 1 : If Puck too close & a very high shooting angle -> slow down, first rotate

                # if (distance_kart_to_puck <= DistanceThreshold) and (np.abs(steering_angle) > ShootingAngleThreshold):
                if print_output:
                    print("Steering Angle: ", steering_angle)
                if distance_kart_to_puck <= DistanceThreshold:

                    acceleration = 0.14
                    brake = False
                    drift = False  # Experiment with drift here

                # For the time being - in all other situations, accelerate till speed limit is hit
                else:
                    speed = np.linalg.norm(self.kart_details[i]['kart']['velocity'])

                    # if self.follow_through_after_reverse[i] <= 0:
                    #     # We're not in follow-through so continue basic operations
                    if distance_kart_to_puck >= 10:
                        if speed <= 14:
                            acceleration = 0.5
                            # acceleration = np.clip((1 - np.abs(steering_angle)) ** 2, 0.15,
                            #                        0.8)  # accelerate basis steering angle
                            brake = False
                            drift = False
                        else:
                            acceleration = 0
                            brake = False
                            drift = False
                    else:
                        if speed <= 10:
                            acceleration = 0.5
                            # acceleration = np.clip((1 - np.abs(steering_angle)) ** 2, 0.075,
                            #                        0.8)  # accelerate basis steering angle
                            brake = False
                            drift = True
                        else:
                            acceleration = 0
                            brake = False
                            drift = True
                    # else:
                    #     print("We're in follow-through, so accelerate at high rate, till puck is found again")
                    #     acceleration = 0.4
                    #     brake = False
                    #     drift = True

                if np.abs(steering_angle) >= max_steer_angle:
                    drift = True
                else:
                    drift = False

                self.kart_actions[i] = {
                    'steer': steering_angle,
                    'acceleration': acceleration,
                    'brake': brake,
                    'drift': drift,
                    'fire': fire_item,
                    'nitro': False,
                    'rescue': rescue
                }

                self.reverse_for_puck_previous_frame[i] = self.reverse_for_puck[i]
                self.puck_in_frame_previous_frame[i] = self.puck_in_frame[i]
                self.puck_3d_coordinates_previous_frame[i] = self.puck_3d_coordinates[i]
                self.total_frame[i] += 1

            # Hold out the Kart - 0 towards back for initial frames, to counteract the puck coming to our own side.
            if self.total_frame[i] <= starting_frames:
                if i == 0:
                    if print_output:
                        print("Do nothing, just wait")
                    self.kart_actions[i] = {
                        'steer': 0,
                        'acceleration': 0.3,
                        'brake': False,
                        'drift': False,
                        'fire': False,
                        'nitro': False,
                        'rescue': False
                    }
                else:
                    if print_output:
                        print("Zoom!")
                    self.kart_actions[i] = {
                        'steer': 0,
                        'acceleration': 1.0,
                        'brake': False,
                        'drift': False,
                        'fire': False,
                        'nitro': True,
                        'rescue': False
                    }

            if print_output:
                print("Final Action - ", self.kart_actions[i])
                print()

        # save_data_to_pickle('/content/DeepLearningFinalProject/state_agent/angle.pkl', angle_details)
        return [self.kart_actions[0], self.kart_actions[1]]
