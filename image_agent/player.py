import numpy as np
import torch
from PIL import Image
from os import path
from torchvision import transforms
import sys

# Import model objects here as well once created

if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def screen_to_world_coordinates(puck_screen_coordinates, proj, view, normalization_factor, image_dimensions):
    WH2 = np.array([image_dimensions[1], image_dimensions[0]]) / 2

    normalized_puck_screen_coordinates = puck_screen_coordinates
    normalized_puck_screen_coordinates[0] = normalized_puck_screen_coordinates[0] / WH2[0] - 1
    normalized_puck_screen_coordinates[1] = normalized_puck_screen_coordinates[1] / WH2[1] - 1

    puck_world_coordinates = np.linalg.inv(view.T) @ np.linalg.inv(proj) @ np.array(list(
        [normalization_factor * normalized_puck_screen_coordinates[0],
         -normalization_factor * normalized_puck_screen_coordinates[1]]) + [normalization_factor,
                                                                            1 - normalization_factor])

    return puck_world_coordinates


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

        self.total_frame = {0: 0, 1: 0}

        self.kart_image = {}

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
        self.model_puck_classifier = torch.load('/Users/adityajariwala/Documents/GitHub/dl-final-project/image_agent/model_puck_final.pt',
                                                map_location="cpu").to(device)
        self.model_puck_classifier.eval()

        print(self.model_puck_classifier)

        self.model_puck_unified = torch.load('/Users/adityajariwala/Documents/GitHub/dl-final-project/image_agent/model_unified_50.pt',
                                             map_location="cpu").to(device)
        self.model_puck_unified.eval()

        print(self.model_puck_classifier)
        print(self.model_puck_unified)

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
        return ['beastie'] * num_players

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
        FollowThroughFrames = 15
        FramesDownTheLine = 3

        all_goals = np.array([[0, 75], [0, -75]])
        own_goal = all_goals[self.team - 1]
        opponents_goal = all_goals[self.team]
        own_corners = np.array([[50, all_goals[self.team - 1][1]], [-50, all_goals[self.team - 1][1]]])

        def world_to_screen_coordinates_one_shot(puck_world_coordinates, proj, view):
            p = proj @ view.T @ np.array(list(puck_world_coordinates) + [1])

            normalized_puck_2d_coordinates = np.array([p[0] / p[2], -p[1] / p[2]])
            normalization_factor = p[2]

            WH2 = np.array([400, 300]) / 2
            puck_2d_coordinates = WH2 * (normalized_puck_2d_coordinates + 1)

            if (np.abs(normalized_puck_2d_coordinates[0]) > 1) or (np.abs(normalized_puck_2d_coordinates[1]) > 1):
                print("Puck's 2-D Coordinates out of Frame")
                puck_found = False
            else:
                print("Puck's 2-D Coordinates in Frame")
                puck_found = True

            if (puck_found == True) and (normalization_factor < 0):
                puck_found = False

            return puck_found, normalization_factor

        def world_to_screen_coordinates(puck_world_coordinates, proj, view):
            p = proj @ view.T @ np.array(list(puck_world_coordinates) + [1])
            normalized_puck_2d_coordinates = np.array([p[0] / p[2], -p[1] / p[2]])
            return normalized_puck_2d_coordinates

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

        print("Team - ", self.team)
        # Loop through both Karts
        for i in [0, 1]:

            print("Actions for Kart -", i)

            self.kart_details[i] = player_state[i]
            self.kart_image[i] = player_image[i]

            proj = self.kart_details[i]['camera']['projection']
            view = self.kart_details[i]['camera']['view']

            # Model to Predict Puck's 3D Coordinates -
            prob_puck_in_frame = self.model_puck_classifier.forward(
                self.transform(Image.fromarray(self.kart_image[i]))[None, :, :, :].to(device))
            self.puck_in_frame[i] = torch.argmax(prob_puck_in_frame) == 1
            self.puck_in_frame[i] = self.puck_in_frame[i].cpu().detach().numpy().copy()

            if self.puck_in_frame[i] == True:
                puck_screen_coordinates, normalization_factor = self.model_puck_unified.forward(
                    self.transform(Image.fromarray(self.kart_image[i]))[None, :, :, :].to(device))
                puck_screen_coordinates = puck_screen_coordinates.cpu().detach().numpy()[0].copy()

                normalized_puck_screen_coordinates = puck_screen_coordinates.copy()
                normalized_puck_screen_coordinates[0] = normalized_puck_screen_coordinates[0] / 200 - 1
                normalized_puck_screen_coordinates[1] = normalized_puck_screen_coordinates[1] / 150 - 1

                print("Predicted Screen Coordinates - ", puck_screen_coordinates, normalized_puck_screen_coordinates)

                normalization_factor = normalization_factor.cpu().detach().numpy()[0][0].copy()
                print("Normalized Factor (Distance)", normalization_factor)

                puck_world_coordinates = screen_to_world_coordinates(puck_screen_coordinates, proj, view,
                                                                     normalization_factor)
                print("Derived 3D Coordinates", puck_world_coordinates)

            else:
                puck_world_coordinates = np.array([0, 0, 0])
                normalized_puck_screen_coordinates = np.array([0, 0])
                normalization_factor = 0

            def world_to_screen_coordinates(puck_world_coordinates, proj, view):
                p = proj @ view.T @ np.array(list(puck_world_coordinates) + [1])
                normalized_puck_2d_coordinates = np.array([p[0] / p[2], -p[1] / p[2]])
                return normalized_puck_2d_coordinates

            ##### Check whether FOLLOW-THROUGH is needed or not ####
            if self.follow_through_after_reverse[i] == 0:

                # We're not in follow-through & need to continue with original action
                print("No follow-through detected for kart!")

            elif self.follow_through_after_reverse[i] > 0:

                print("Kart in follow-through!")
                # We're in follow-through, so we need to treat like the puck is already present in frame, at last spotted location
                self.puck_in_frame[i] = self.puck_in_frame_previous_frame[i]
                puck_world_coordinates = self.puck_3d_coordinates_previous_frame[i]

                # We also reduce the follow-through frame count, as the kart goes through follow-through for current frame
                self.follow_through_after_reverse[i] -= 1

            else:

                # Bug - we've hit a negative value on follow-through frames, so we reset to 0
                self.follow_through_after_reverse[i] = 0

                #### End of FOLLOW-THROUGH ####

            # If puck in frame, then we proceed further, otherwise we set actions to reverse, and steer in left
            if self.puck_in_frame[i] == False:

                print('Puck Not in Frame - 1. Reversing..')
                self.reverse_for_puck[i] = True

                if self.reverse_for_puck_previous_frame[i] == True:
                    # We've been reversing since previous frame as well
                    # So, no need to update follow-through
                    print("Continuing to Reverse..")
                else:
                    # We're beginning to reverse now
                    # print("Starting to Reverse for first time..")

                    self.puck_screen_coordinates[i] = np.array([0, 0])
                    self.puck_3d_coordinates[i] = [0, 0]

                # Instead of rotating, we try reversing towards our goal, facing opponent's goal
                self.kart_actions[i] = {
                    'steer': -1.0,
                    'acceleration': 0,
                    'brake': True,
                    'drift': False,
                    'nitro': False,
                    'rescue': False
                }
                continue

            else:

                print('Puck in Frame. Setting Aim and Chasing')
                self.reverse_for_puck[i] = False

                # Reset puck's utilization basis partner's puck location
                self.utilize_partners_puck_location[i] = False

                # Check if the puck was there in previous frame
                if self.reverse_for_puck_previous_frame[i] == True:
                    # We've just found the puck in this frame, then we start the follow-through count
                    self.follow_through_after_reverse[
                        i] = FollowThroughFrames  # Follow-Through for 5 frames

                # Model to predict the location of puck in frame (pixel corrdinates)
                self.puck_screen_coordinates[i] = np.array([0, 0])
                self.puck_3d_coordinates[i] = puck_world_coordinates.copy()
                self.normalized_puck_screen_coordinates[i] = normalized_puck_screen_coordinates.copy()

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

                ### Implementation of Actions -

                ## a) Shooting Direction
                # Case 1 : Puck far away from Kart -> Shoot towards the puck directly
                if distance_kart_to_puck >= DistanceThreshold:

                    # To prevent glitching as we approach the puck
                    angle_shooting_direction = np.degrees(np.arccos(np.dot(kart_direction,
                                                                           kart_to_puck_direction)))  # Both vectors have a norm 1, so dot represent cos(theta)
                    direction_of_steering = np.sign(np.cross(kart_direction, kart_to_puck_direction))
                    angle_shooting_direction = -direction_of_steering * angle_shooting_direction
                    steering_angle = (angle_shooting_direction) / 90.
                    steering_angle = np.clip(steering_angle, -1, 1)

                    if np.abs(steering_angle) >= 0.1:
                        # The shooting direction is simple the angle between kart's orientation vector and vector from kart's back to puck
                        angle_shooting_direction = np.degrees(np.arccos(np.dot(kart_direction,
                                                                               new_kart_to_puck_direction)))  # Both vectors have a norm 1, so dot represent cos(theta)
                        direction_of_steering = np.sign(np.cross(kart_direction, new_kart_to_puck_direction))
                        angle_shooting_direction = -direction_of_steering * angle_shooting_direction

                        print("Case 1")
                        print('Steering Angle - ', angle_shooting_direction)

                        steering_angle = (angle_shooting_direction) / 90.
                        steering_angle = np.clip(steering_angle, -1, 1)

                else:

                    ### Another Team's implementation ####
                    # puck_y_coordinate = normalized_puck_screen_coordinates[1]
                    # direction_of_steering_to_goal = np.sign(np.cross(kart_direction, kart_to_goal_direction))
                    # angle_of_steering_to_goal = np.arccos(np.dot(kart_direction, kart_to_goal_direction))
                    # angle_of_steering_to_goal = -direction_of_steering_to_goal*angle_of_steering_to_goal

                    # norm_goal_distance = ((np.clip(distance_kart_to_opponent_goal, 10, 100) - 10) / 90) + 1
                    # distance = 1 / norm_goal_distance ** 3
                    # aim_point = puck_y_coordinate + np.sign(puck_y_coordinate - angle_of_steering_to_goal) * 0.3 * distance
                    # steering_angle = np.clip(aim_point * 15, -1, 1)

                    # Then, we check if the goal and puck are on different sides of kart
                    direction_of_steering_to_goal = np.sign(np.cross(kart_direction, kart_to_goal_direction))
                    direction_of_steering_to_puck = np.sign(np.cross(kart_direction, new_kart_to_puck_direction))

                    angle_of_steering_to_goal = np.arccos(np.dot(kart_direction, kart_to_goal_direction))
                    angle_of_steering_to_puck = np.arccos(np.dot(kart_direction, new_kart_to_puck_direction))

                    if direction_of_steering_to_goal * direction_of_steering_to_puck < 0:
                        # It indicates that movings towards goal might move away from puck, so first we get closer to puck till these angles align
                        angle_shooting_direction = np.degrees(np.arccos(np.dot(kart_direction,
                                                                               new_kart_to_puck_direction)))  # Both vectors have a norm 1, so dot represent cos(theta)
                        direction_of_steering = np.sign(np.cross(kart_direction, new_kart_to_puck_direction))
                        angle_shooting_direction = -direction_of_steering * angle_shooting_direction

                    else:
                        angle_shooting_direction_puck = np.degrees(np.arccos(np.dot(kart_direction,
                                                                                    new_kart_to_puck_direction)))  # Both vectors have a norm 1, so dot represent cos(theta)
                        direction_of_steering_puck = np.sign(np.cross(kart_direction, new_kart_to_puck_direction))
                        angle_shooting_direction_puck = -direction_of_steering_puck * angle_shooting_direction_puck

                        angle_shooting_direction_goal = np.degrees(np.arccos(np.dot(kart_direction,
                                                                                    kart_to_goal_direction)))  # Both vectors have a norm 1, so dot represent cos(theta)
                        direction_of_steering_goal = np.sign(np.cross(kart_direction, kart_to_goal_direction))
                        angle_shooting_direction_goal = -direction_of_steering_goal * angle_shooting_direction_goal

                        # weight = np.clip(distance_kart_to_opponent_goal, 1, 100)/100
                        # angle_shooting_direction = weight*angle_shooting_direction_puck + (1-weight)*angle_shooting_direction_goal

                        angle_shooting_direction = angle_shooting_direction_goal

                    print("New Direction (Angle) -", angle_shooting_direction)
                    # print("New Direction (Velocity) -",angle_shooting_direction_velocity)

                    print("Case 2")
                    print('Steering Angle - ', angle_shooting_direction)

                    # Steering Angle - Treat these angles as 3X more steering
                    steering_angle = 3 * (angle_shooting_direction) / 90.
                    steering_angle = np.clip(steering_angle, -1, 1)

                ## b) Accelerate/Brake
                # Case 1 : If Puck too close & a very high shooting angle -> slow down, first rotate

                # if (distance_kart_to_puck <= DistanceThreshold) and (np.abs(steering_angle) > ShootingAngleThreshold):
                if (distance_kart_to_puck <= DistanceThreshold):

                    acceleration = 0.14
                    brake = False
                    drift = True  # Experiment with drift here

                # For the time being - in all other situations, accelerate till speed limit is hit
                else:
                    speed = np.linalg.norm(self.kart_details[i]['kart']['velocity'])

                    if self.follow_through_after_reverse[i] <= 0:
                        # We're not in follow-through so continue basic operations
                        if (distance_kart_to_puck >= 10):
                            if speed <= 12:
                                acceleration = np.clip((1 - np.abs(steering_angle)) ** 2, 0.15,
                                                       0.8)  # accelerate basis steering angle
                                brake = False
                                drift = False
                            else:
                                acceleration = 0
                                brake = False
                                drift = False
                        else:
                            if speed <= 8:
                                acceleration = np.clip((1 - np.abs(steering_angle)) ** 2, 0.075,
                                                       0.8)  # accelerate basis steering angle
                                brake = False
                                drift = False
                            else:
                                acceleration = 0
                                brake = False
                                drift = False
                    else:
                        print("We're in follow-through, so accelerate at high rate, till puck is found again")
                        acceleration = 0.4
                        brake = False
                        drift = True

                if np.abs(steering_angle) >= 0.4:
                    drift = True
                else:
                    drift = False

                self.kart_actions[i] = {
                    'steer': steering_angle,
                    'acceleration': acceleration,
                    'brake': brake,
                    'drift': drift,
                    'nitro': False,
                    'rescue': False
                }

                self.reverse_for_puck_previous_frame[i] = self.reverse_for_puck[i]
                self.puck_in_frame_previous_frame[i] = self.puck_in_frame[i]
                self.puck_3d_coordinates_previous_frame[i] = self.puck_3d_coordinates[i]
                self.total_frame[i] += 1

            # Hold out the Kart - 0 towards back for initial frames, to counteract the puck coming to our own side.
            if i == 0:
                if (self.total_frame[i] <= 40):
                    print("Do nothing, just wait")
                    self.kart_actions[i] = {
                        'steer': 0,
                        'acceleration': 0,
                        'brake': False,
                        'drift': False,
                        'nitro': False,
                        'rescue': False
                    }

            print("Final Action - ", self.kart_actions[i])

        # save_data_to_pickle('/content/DeepLearningFinalProject/state_agent/angle.pkl', angle_details)
        return [self.kart_actions[0], self.kart_actions[1]]
