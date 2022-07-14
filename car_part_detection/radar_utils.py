import csv
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def pick_biggest(region_id, how_many, original_list):
    '''
    Pick the biggest n regions

    Original list: list of regions in which each region is represented as
    a list like [region_id, x centre, y-center, width, height]
    '''
    final_list = []
    for l in original_list:
        if l[0] == region_id:
            # append area
            l.append(l[3] * l[4])
            final_list.append(l)
    final_list.append([region_id, 0, 0, 0.010, 0.010, 0.01*0.01]) #Append a dummy region in case there are no regions
    final_list.sort(key=lambda x:x[-1], reverse=True)
    final_list = final_list[:how_many]
    return final_list

def append_area(region):
    # pass if area is already there
    if len(region) > 5:
        pass
    # otherwise, append area as 5th element of the list
    else:
        region.append(region[-2] * region[-1])
    return region

def front_driverside_regions_img(txt_file_path):
    '''
    Inputs: path to output txt file of yolov5 containing detected regions
    Output: list of picked regions (ie the biggest light, wheel/door/sideglass
    to its right, and glass (windshield) above and closest to light). Each region
    is a list consisting of [region number, xcentre, ycentre, width, height, area].
    Region numbers represent the following:
    - 0 Light
    - 1 Wheel
    - 2 Glass
    - 3 Door
    - 4 Sideglass
    '''
    regions = []
    with open(txt_file_path) as f:
        csv_reader = csv.reader(f, delimiter=' ')
        for row in csv_reader:
            row = [float(i) for i in row]
            regions.append(row)
    
    light = pick_biggest(0, 1, regions)[0]
    
    wheels = [region for region in regions if region[0] == 1]
    wheels = [wheel for wheel in wheels if wheel[1] > light[1]] # pick wheels right of light
    if len(wheels) < 1: #if there are no wheels to the right of light
        wheel = [1, 0, 0, 0.075, 0.075]
    else:
        wheel = min(wheels, key=lambda x:x[1]) # leftmost among those wheels (closest to light)
        
    glasses = [region for region in regions if region[0] == 2]
    glasses = [glass for glass in glasses if glass[2] < light[2]] # pick glasses above light
    if len(glasses) < 1:
        glass = [2, 0, 0, 0.075, 0.075]
    else:
        glass = min(glasses, key=lambda x:np.abs(x[2]-light[2])) # glass closest to light in y-axis
    
    doors = [region for region in regions if region[0] == 3]
    doors = [door for door in doors if door[1] > light[1]]
    if len(doors) < 1:
        door = [3, 0, 0, 0.075, 0.075]
    else:
        door = min(doors, key=lambda x:x[1])

    sglasses = [region for region in regions if region[0] == 4]
    sglasses = [sglass for sglass in sglasses if sglass[1] > light[1]]
    if len(sglasses) < 1:
        sglass = [4, 0, 0, 0.075, 0.075]
    else:
        sglass = min(sglasses, key=lambda x:x[1])


    
    light = append_area(light)
    wheel = append_area(wheel)
    door = append_area(door)
    sglass = append_area(sglass)
    glass = append_area(glass)
    
    return light, wheel, glass, door, sglass


def front_driverside_regions_vid(txt_file_path):
    '''
    Same as front_driverside_regions_img but does not do anything when no
    regions are detected.

    Inputs: path to output txt file of yolov5 containing detected regions
    Output: list of picked regions (ie the biggest light, wheel/door/sideglass
    to its right, and glass (windshield) above and closest to light). Each region
    is a list consisting of [region number, xcentre, ycentre, width, height, area].
    Region numbers represent the following:
    - 0 Light
    - 1 Wheel
    - 2 Glass
    - 3 Door
    - 4 Sideglass
    '''
    # Default values (used when there are no detected regions of each kind)
    wheel = [2, 0, 0, 0.075, 0.075]
    glass = [3, 0, 0, 0.075, 0.075]
    door = [4, 0, 0, 0.05, 0.05]
    sglass = [5, 0, 0, 0.035, 0.035]

    # open the text file in txt_file_path and put the contents into a list (regions)
    regions = []
    with open(txt_file_path) as f:
        csv_reader = csv.reader(f, delimiter=' ')
        for row in csv_reader:
            row = [float(i) for i in row]
            regions.append(row)
    
    # The light with largest area serves as the reference point for the other regions
    light = pick_biggest(0, 1, regions)[0]
    
    wheels = [region for region in regions if region[0] == 1]
    wheels = [wheel for wheel in wheels if wheel[1] > light[1]] # pick wheels right of light
    if len(wheels)>0:
        wheel = min(wheels, key=lambda x:x[1]) # leftmost among those wheels (closest to light)
        
    glasses = [region for region in regions if region[0] == 2]
    glasses = [glass for glass in glasses if glass[2] < light[2]] # pick glasses above light
    if len(glasses)>0:
        glass = min(glasses, key=lambda x:np.abs(x[1]-light[1])) # glass closest to light in x-axis
    
    doors = [region for region in regions if region[0] == 3]
    doors = [door for door in doors if door[1] > light[1]]
    if len(doors)>0:
        door = min(doors, key=lambda x:x[1])

    sglasses = [region for region in regions if region[0] == 4]
    sglasses = [sglass for sglass in sglasses if sglass[1] > light[1]]
    if len(sglasses)>0:
        sglass = min(sglasses, key=lambda x:x[1])
    
    light = append_area(light)
    wheel = append_area(wheel)
    door = append_area(door)
    sglass = append_area(sglass)
    glass = append_area(glass)
    
    return light, wheel, glass, door, sglass


def radar_chart(light, wheel, glass, door, sglass):
    """
    Inputs: Region lists for light, wheel, glass, door, sideglass
    Output: Radar chart
    """
    
    IDEAL_LIGHT_AREA = 0.03
    IDEAL_WHEEL_AREA = 0.07
    IDEAL_GLASS_AREA = 0.09
    IDEAL_DOOR_AREA = 0.04
    IDEAL_SGLASS_AREA = 0.0075
    
    light_score = light[5] / IDEAL_LIGHT_AREA * 100
    wheel_score = wheel[5] / IDEAL_WHEEL_AREA * 100
    glass_score = glass[5] / IDEAL_GLASS_AREA * 100
    door_score = door[5] / IDEAL_DOOR_AREA * 100
    sglass_score = sglass[5] / IDEAL_SGLASS_AREA * 100

    labels = ['Light', 'Wheel', 'Glass', 'Door', 'Sideglass']
    values = [light_score, wheel_score, glass_score, door_score, sglass_score]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()

    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.plot(angles, values, color='#1aaf6c', linewidth=1)
    ax.fill(angles, values, color='#1aaf6c', alpha=0.25)
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    ax.set_thetagrids(np.degrees(angles), labels)

    # adjust label locations so they don't overlap with chart itself
    for label, angle in zip(ax.get_xticklabels(), angles):
        if angle in (0, np.pi):
            label.set_horizontalalignment('center')
        elif 0 < angle < np.pi:
            label.set_horizontalalignment('left')
        else:
            label.set_horizontalalignment('right')

    # Make sure grid labels go from 0 to 100
    ax.set_ylim(0, 100)
    # You can also set gridlines manually like this:
    # ax.set_grids([20, 40, 60, 80, 100])

    # Set position of y-labels (0-100) to be in the middle
    # of the first two axes
    ax.set_rlabel_position(180/len(labels))

    # MORE STYLING
    # Chagne colour of tick labels
    ax.tick_params(colors='#222222')
    # Make y-axis (0-100) labels smaller
    ax.tick_params(axis='y', labelsize=8)
    # Change colour of circular gridlines
    ax.grid(color='#AAAAAA')
    # Change colour of outermost gridline
    ax.spines['polar'].set_color('#222222')
    # Change background colour of inside the circle
    ax.set_facecolor('#FAFAFA')

    # Give the chart a title and some padding above 'light' label
    # ax.set_title('Picture Quality Criterion', y=1.08)
    plt.show()
