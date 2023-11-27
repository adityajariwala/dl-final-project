import numpy as np
import math
import torch

class Team:
    agent_type = 'image'

    def __init__(self):
        """
          TODO: Load your agent here. Load network parameters, and other parts of our model
          We will call this function with default arguments only
        """
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
        self.goal = [0, 75] if team % 2 == 0 else [0, -75]
        self.own_goal = [0, 75] if team % 2 == 1 else [0, -75]
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
        
        model_output = [True, (1, 2), 1.0]

        distance_to_puck = model_output[2]
        puck_location = model_output[1]
        puck_onscreen = model_output[0]
        res = []

        for p_state in player_state:
          player = dict()
          front = p_state['kart']['front'][[0,2]]
          curr_location = p_state['kart']['location'][[0,2]]
          kart_direction = ((front-curr_location) / torch.norm(front-curr_location)).numpy()

          kart_x_stuck_on_edge = np.abs(front[0]) > 18
          kart_y_stuck_on_edge = np.abs(front[1]) > 54

          kart_x_perp = np.abs(kart_direction[0]) > 0.85
          kart_stuck_x = kart_x_stuck_on_edge and kart_x_perp

          kart_y_perp = np.abs(kart_direction[1]) > 0.85
          kart_stuck_y = kart_y_stuck_on_edge and kart_y_perp

          goal_direction = ((torch.tensor(self.goal) - front) / torch.norm(torch.tensor(self.goal) - front)).numpy()
          kart_direction = ((front - curr_location) / torch.norm(front - curr_location)).numpy()
          own_goal_direction = ((torch.tensor(self.own_goal) - front) / torch.norm(torch.tensor(self.own_goal) - front)).numpy()
          theta_kart_to_goal = np.arctan2(self.goal[1] - curr_location[1], self.goal[0] - curr_location[0])


          if not puck_onscreen or distance_to_puck > 5:
              print('kart stuck somewhere')
              if kart_stuck_x or kart_stuck_y:
                  player['acceleration'] = 1
                  player['brake'] = False
                  player['steer'] = 0.5 * np.random.rand()

          else:
              print('puck onscreen')
              # not sure how exactly distances will work, but assuming that it is possible that we are slightly ahead of puck but it is still visible 
              # hence the negative distance value
              if distance_to_puck < 0.25 and distance_to_puck > -0.15:
                  if np.sign(goal_direction[1]) == np.sign(kart_direction[1]) and theta_kart_to_goal >15:
                      shooting_direction = np.array(puck_location) - np.array(curr_location)
                      shooting_direction_normalized = 0.2 * shooting_direction / np.linalg.norm(shooting_direction)
                      theta_kart_to_puck = np.degrees(np.arctan2(shooting_direction_normalized[1], shooting_direction_normalized[0]))
                      v_puck_to_goal = np.array([self.goal[0] - puck_location[0], self.goal[1] - puck_location[1]])
                      theta_puck_to_own_goal = np.degrees(np.arctan2(self.own_goal[0], self.own_goal[1]))

                      theta_puck_to_goal = np.degrees(np.arctan2(v_puck_to_goal[1], v_puck_to_goal[0]))
                      required_angle_of_deviation_in_puck = theta_puck_to_goal - theta_kart_to_puck
                      radius_puck = 0.1
                      deflection_angle = 180./np.pi*np.arctan2(radius_puck*np.sin(np.radians(required_angle_of_deviation_in_puck)), distance_to_puck - radius_puck*np.cos(np.radians(required_angle_of_deviation_in_puck)))
                      updated_shooting_direction_angle = theta_kart_to_puck - (deflection_angle)

                      approach_vs_opp_goal = updated_shooting_direction_angle - theta_puck_to_goal
                      approach_vs_own_goal = updated_shooting_direction_angle - theta_puck_to_own_goal

                      steering_angle = theta_kart_to_goal - updated_shooting_direction_angle
                      if steering_angle > 180:
                          steering_angle = steering_angle - 360
                      
                      if steering_angle < -180:
                          steering_angle = 360 + steering_angle

                      player['steer'] = steering_angle
                      # need to steer away if the we are going towards our own goal
                  elif np.sign(front[1]) == np.sign(self.own_goal[1]) and np.sign(front[1]) == np.sign(kart_direction[1]) and np.abs(front[1]) > 15:
                      goal_displacement = np.sign(kart_direction[1])*(kart_direction[0] - own_goal_direction[0])
                      # TODO figure this part out
                      # steer = goal_displacement / np.abs(distance_to_puck)
                      # player['steer'] 
          res.append[player]
          
        return res
        # TODO: Change me. I'm just cruising straight
        # return [dict(acceleration=1, steer=0)] * self.num_players
