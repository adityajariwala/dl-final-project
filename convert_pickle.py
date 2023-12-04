import pickle
from os import path
import numpy as np


def world_to_screen_coordinates_one_shot(puck_world_coordinates, proj, view, image_dimensions):
    p = proj @ view.T @ np.array(list(puck_world_coordinates) + [1])
    # basic_output = p.copy()
    normalized_puck_2d_coordinates = np.array([p[0] / p[2], -p[1] / p[2]])
    WH2 = np.array([image_dimensions[1], image_dimensions[0]])/2
    puck_2d_coordinates = WH2*(normalized_puck_2d_coordinates+1)
    if (np.abs(normalized_puck_2d_coordinates[0]) > 1) | (np.abs(normalized_puck_2d_coordinates[1]) > 1):
        puck_found = False
    else:
        puck_found = True
    # Regenerate the puck coordinates back
    distance = p[2]
    x = np.linalg.inv(view.T) @ np.linalg.inv(proj) @ np.array(list([distance*normalized_puck_2d_coordinates[0], -distance*normalized_puck_2d_coordinates[1]]) + [distance,1-distance])
    regen_puck_world_coordinates = x.copy()
    if puck_found:
        error = np.linalg.norm(regen_puck_world_coordinates[[0,2]] - np.array(puck_world_coordinates)[[0,2]])
    else:
        error = 0
    return puck_found, distance, puck_2d_coordinates, error


# files = ['image_v_geoffrey.pkl']
files = ['image_v_ai.pkl', 'image_v_geoffrey.pkl', 'image_v_jurgen.pkl', 'image_v_yann.pkl',
         'image_v_yoshua.pkl']
data = []

for file in files:
    with open(path.join(path.dirname(path.abspath(__file__)), file), 'rb') as fr:
        try:
            count = 0
            while True:
                data.append(pickle.load(fr))
                count = count + 1
        except EOFError:
            pass

for i in range(len(data)):
    for team in ['team1', 'team2']:
        for kart in [0, 1]:
            puck_world_coordinates = data[i]['soccer_state']['ball']['location'].copy()
            proj = data[i][team + '_state'][kart]['camera']['projection'].copy()
            view = data[i][team + '_state'][kart]['camera']['view'].copy()
            image = data[i][team + '_images'][kart].copy()
            image_dimensions = image.shape
            puck_found, distance, puck_2d_coordinates, _ = world_to_screen_coordinates_one_shot(puck_world_coordinates,
                                                                                                proj, view,
                                                                                                image_dimensions)
            # This is a bug observed in certain cases -
            # when the puck is not actually present, but it is shown in frame
            if distance < 0:
                puck_found = False
            data[i][team + "_kart_" + str(kart) + "_puck_screen_location"] = [puck_found, puck_2d_coordinates, distance]

with open(path.join(path.dirname(path.abspath(__file__)), "big_data_5k.pkl"), 'wb') as f:
    pickle.dump(data, f)
