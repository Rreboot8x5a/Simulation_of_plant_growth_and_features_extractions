"""
Copyright 2020 Quaranta Roberto

This file is part of Simulation_of_plant_growth_and_features_extractions.

Simulation_of_plant_growth_and_features_extractions is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Simulation_of_plant_growth_and_features_extractions is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Simulation_of_plant_growth_and_features_extractions.  If not, see <https://www.gnu.org/licenses/>.
"""
"""
This script implements the phenotyping of the basil plant through its growing.
"""
from Clustering import *

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import math
import matplotlib.pyplot as plt
from Get_real_values import read_real_parameters_file


def plot_leaf_number_evolution(plantAllTime, time):
    # If we want to compare real values and retrieved values, we have to use 2 times this function
    points = np.zeros((time - 1, 2))  # The data that will be plot
    points[:, 0] = np.arange(1, len(points) + 1)  # The x axis is just 1, 2, 3, ..., time
    for t in plantAllTime:
        nb_leaves = len(plantAllTime[t]) - 1  # The plant dictionary for a precise time is just : height, leaf 1, leaf 2, ...
        points[t - 1, 1] = nb_leaves
    plt.plot(points[:, 0], points[:, 1])
    plt.ylabel('Number of leaves')
    plt.xlabel('Time')
    plt.title('Evolution of leaf number for basil plant')
    # plt.show()


def plot_height_evolution(plantAllTime, time):
    # If we want to compare real values and retrieved values, we have to use 2 times this function
    points = np.zeros((time - 1, 2))  # The data that will be plot
    points[:, 0] = np.arange(1, len(points) + 1)  # The x axis is just 1, 2, 3, ..., time
    for t in plantAllTime:
        points[t - 1, 1] = plantAllTime[t]['Height']
    plt.plot(points[:, 0], points[:, 1])
    plt.ylabel('Height (cm)')
    plt.xlabel('Time')
    plt.title('Evolution of height for basil plant')
    # plt.show()


def plot_leaf_evolution(plantAllTime, time, characteristic, leafNumber):
    # If we want to compare real values and retrieved values, we have to use 2 times this function
    points = np.zeros((time - 1, 2))  # The data that will be plot
    points[:, 0] = np.arange(1, len(points) + 1)  # The x axis is just 1, 2, 3, ..., time
    for t in plantAllTime:
        try:
            leaf = plantAllTime[t][leafNumber]
        except:
            leaf = {
                "Leaf length": 0,
                "Width": 0,
                "Angle": 0,
                "Node height": 0,
                "Hidden": 0,
                "Main axis": 0,
                "Total length": 0,
                "Quarter": 0
            }
        points[t - 1, 1] = leaf[characteristic]
    plt.plot(points[:, 0], points[:, 1])
    plt.ylabel(characteristic)
    plt.xlabel('Time')
    plt.title('Evolution of {} for leaf {} basil plant'.format(characteristic, leafNumber))


def evolution(time):
    """
    This function will read several txt files, the files with the ray tracing data in the BasilData2 folder and the
    Real_parameters_of_basil2.txt containing the data extract from the plant object of CPlantBox
    """
    # This function is especially made for the basil plant, the difficulty about this part was retrieving
    # a specific leaf between 2 time steps. To understand what was done, please refer the word files sent by email
    plantAllTime = {}
    firstTry = True
    for dt in range(1, time):
        ncoord = read_data(dt, 'basil2')

        # Create the dictionary with the key parameter
        plantChar = {
            "Height": 0
        }

        totalHeight = get_height(ncoord)
        plantChar["Height"] = totalHeight

        # Change data format and using DBSCAN to get clusters
        leaves, nb_clusters, predictions = get_leaves(ncoord)

        if len(leaves) != 0:

            # TODO : implement the main axes retrieving at the first time steps (actually the main axes are defined
            #  because I checked the final plot and compute manually the axes before launching this function)
            # Main axes for BasilData
            mainAxis1 = [-0.46, -2.95, 0]
            mainAxis2 = [2.95, -0.46, 0]

            # Main axes for BasilData2
            mainAxis1 = [0.888, 2.6385, 0]
            mainAxis2 = [2.6385, -0.888, 0]

            leaves = separate_united_leaves(leaves)
            leaves, centerLeaves = unite_cluster(leaves, mainAxis1, mainAxis2)

            if firstTry:
                for i in range(len(leaves)):
                    # TODO Add security tests if one or several leaves are missing after clustering
                    leafLength, width, angle, zNode, leafMainAxis, length = get_parameters(leaves[i])
                    isHidden = False
                    leafDict = {
                        "Leaf length": leafLength,
                        "Width": width,
                        "Angle": angle,
                        "Node height": zNode,
                        "Hidden": isHidden,
                        "Main axis": leafMainAxis,
                        "Total length": length,
                        "Quarter": centerLeaves[i]
                    }
                    if i == 0:
                        for j in range(len(leaves)):
                            plantChar[j + 1] = {}
                            keyList = list(plantChar.keys())
                    plantChar[keyList[-1] - i] = leafDict
                    firstTry = False
            else:
                oldPlant = plantAllTime.get(dt-1)
                oldKeyList = list(oldPlant)
                newFloor = False
                for i in range(len(leaves)):
                    if len(leaves[i]) > 1:
                        if oldPlant != None:
                            oldKeyList = list(oldPlant.keys())
                        if i < 4:  # The leaf cannot be hidden
                            leafLength, width, angle, zNode, leafMainAxis, length = get_parameters(leaves[i])
                            isHidden = False
                        else:
                            width, angle, zNode, leafMainAxis, length = get_hidden_leaf_characteristics(leaves[i])
                            leafLength = 0
                            isHidden = True

                        try:
                            quarter = centerLeaves[i]
                        except:
                            quarter = 0
                        leafDict = {
                            "Leaf length": leafLength,
                            "Width": width,
                            "Angle": angle,
                            "Node height": zNode,
                            "Hidden": isHidden,
                            "Main axis": leafMainAxis,
                            "Total length": length,
                            "Quarter": quarter
                        }
                        if i < 1:  # The first leaf in the list (the upper one)
                            if (oldPlant[oldKeyList[-1]]['Quarter'] - centerLeaves[i]) % 2 == 1:
                                for j in range(oldKeyList[-1] + 2):  # There are 2 more leaves than before
                                    plantChar[j + 1] = {}
                                actualKeyList = list(plantChar.keys())
                                newFloor = True
                            else:  # The number of leaves remains the same as before
                                for j in range(oldKeyList[-1]):
                                    plantChar[j + 1] = {}
                                actualKeyList = list(plantChar.keys())

                        gap = 0
                        if newFloor == True:
                            gap = 2  # If there is a new floor, the end of the old list is 2 units lesser than the new one
                        if i < 2 and newFloor == True:
                            plantChar[actualKeyList[-1] - i] = leafDict
                        else:
                            # The following if else are there to make equal the leaves between two step times
                            if leafDict['Quarter'] == oldPlant[oldKeyList[-1] - i + gap]['Quarter']:
                                plantChar[actualKeyList[-1] - i] = leafDict
                                if isHidden == True:
                                    plantChar[actualKeyList[-1] - i]['Leaf length'] = oldPlant[oldKeyList[-1] - i + gap]['Leaf length']
                            else:
                                # We need to avoid storing a leaf on the wrong floor
                                if i % 2 == 0:  # There are 2 leaves per floor, the first treated is when i is even and the second treated is when i is odd
                                    plantChar[actualKeyList[-1] - i - 1] = leafDict  # The corresponding leaf between t and t-1 are not on the same position
                                    if isHidden == True:
                                        plantChar[actualKeyList[-1] - i - 1]['Leaf length'] = \
                                        plantAllTime[dt - 1][actualKeyList[-1] - i - 1]['Leaf length']
                                        if width < plantAllTime[dt-1][actualKeyList[-1] - i - 1]['Width']:
                                            plantChar[actualKeyList[-1] - i - 1]['Width'] = \
                                                plantAllTime[dt-1][actualKeyList[-1] - i - 1]['Width']
                                        if length < plantAllTime[dt-1][actualKeyList[-1] - i - 1]['Total length']:
                                            plantChar[actualKeyList[-1] - i - 1]['Total length'] = \
                                                plantAllTime[dt - 1][actualKeyList[-1] - i - 1]['Total length']
                                else:  # The leaf is the second one of the floor
                                    plantChar[actualKeyList[-1] - i + 1] = leafDict  # The corresponding leaf between t and t-1 are not on the same position
                                    if isHidden == True:
                                        plantChar[actualKeyList[-1] - i + 1]['Leaf length'] = \
                                        plantAllTime[dt - 1][actualKeyList[-1] - i + 1]['Leaf length']
                                        if width < plantAllTime[dt-1][actualKeyList[-1] - i + 1]['Width']:
                                            plantChar[actualKeyList[-1] - i + 1]['Width'] = \
                                                plantAllTime[dt-1][actualKeyList[-1] - i + 1]['Width']
                                        if length < plantAllTime[dt-1][actualKeyList[-1] - i + 1]['Total length']:
                                            plantChar[actualKeyList[-1] - i + 1]['Total length'] = \
                                                plantAllTime[dt - 1][actualKeyList[-1] - i + 1]['Total length']

            plantAllTime[dt] = plantChar

    # When everything is done, we retrieve the correct values saved in a txt file to plot a comparison
    # between real values and computed values with the nadir view
    real = read_real_parameters_file('Real_parameters_of_basil2.txt')
    real2 = {}
    for param in real:
        if type(param) == str:
            real2[param] = real[param]
        if type(param) != str:
            # As the function evolution create error when a leaf is hidden, we stop the function before the error
            # and so we discard the time steps not measured because of this
            if param < 39:
                real2[param] = real[param]

    plot_leaf_number_evolution(plantAllTime, time)
    plot_leaf_number_evolution(real2, time)
    plt.show()

    plot_height_evolution(plantAllTime, time)
    plot_height_evolution(real2, time)
    plt.show()

    characteristic = 'Node height'
    leafNumber = 3
    plot_leaf_evolution(plantAllTime, time, characteristic, leafNumber)
    plot_leaf_evolution(real2, time, characteristic, leafNumber)
    plt.show()

# evolution(39)
