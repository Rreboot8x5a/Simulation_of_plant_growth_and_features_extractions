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
This script is the one using for testing the code. The only one useful to run (as the other ones are just function script)
It uses the basic function of this work to plot several results.
If you do not want your computer to die because of too much plot windows opened, please use the debug mode to stop the code at certain points
"""
import os
import sys
import vtk
sys.path.append("..")
import plantbox as pb  # CPlantBox Python Binding

sys.path.append("../tutorial")
from jupyter.CPlantBox_PiafMunch import *
from Create_Plants import *
from vtk_plot_bis import *
import math
import matplotlib.pyplot as plt
import numpy as np
from Get_real_values import *
from TimeEvolution import *
# plotly.__version__


time = 50  # how many days the plant need to grow, make it smaller, for example 15 to see if the plant becomes smaller


def for_the_presentation():
    t = 50
    height = 75
    trueLength = height * math.tan(37.5 * math.pi / 180) * 2

    # Plot the nodes of the basil plant object
    basil = create_basil()
    basil.initialize(True)
    basil.simulate(t)
    fig = visual_plant(basil)
    fig.show()

    # Read the txt file with the ray tracing data of the basil plant after 49 days of simulation
    coord_basil = read_data(49, 'basil2')
    plot_data(coord_basil[:-2], coord_basil[:-2, 2])  # The last point is 0,0,0 and the previous one is the camera position
    basil_leaves, nb_classes, pred = get_leaves(coord_basil)
    coord_basil = get_numpy_array(basil_leaves)
    plot_data(coord_basil, coord_basil[:, 3])

    # Plot the nodes of the arabidopsis plant object
    arab = create_arabidopsis()
    arab.initialize(True)
    arab.simulate(t)
    fig = visual_plant(arab)
    fig.show()  # Need to be done several times because of the randomization

    # Read the txt file with the ray tracing data of the arabidopsis plant after 49 days of simulation
    coord_arab = read_data(626, 'arabidopsis')  # The 626 is there only because the txt 626 file is an experimental one
    plot_data(coord_arab[:-2], coord_arab[:-2, 2]) # The last point is 0,0,0 and the previous one is the camera position
    arab_leaves, nb_classes, pred = get_leaves(coord_arab, trueLength/640)
    coord_arab = get_numpy_array(arab_leaves)
    plot_data(coord_arab, coord_arab[:, 3])

    # This function is a bit buggy with the debugging mode of pycharm, so it is put at the end of the function
    # to avoid annoying problem
    plot_plant(basil, "Basil plot", plantType="basil")  # Plot in 3D using VTK
    plot_plant(arab, "Arabidopsis plot", plantType="arabidopsis")


def multiple_height_test():
    # This function was made to plot the results of clustering with different height in the camera position
    resolution = [640, 576]
    height = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125]
    trueLength = []
    nb_points = []

    basilList = []
    basil = create_basil()
    basil.initialize(True)
    for j in range(10):
        basil.simulate(5)
        basilList.append(basil.copy())

    for i in range(len(height)):
        trueLength.append(height[i] * math.tan(37.5 * math.pi / 180) * 2)
    for i in range(len(height)):
        for j in range(len(basilList)):
            coords = camera_view(basilList[j], height=height[i], resolution=resolution, trueLength=trueLength[i],
                                 isPlotted=False, plantType="basil")
            nb_points.append(len(coords))
            file = open("/home/roberto/Téléchargements/CPlantBox-vierge/Height/basil_coordinates_height{}_t{}.txt".format(height[i], (j + 1) * 5), "w")
            for coord in coords:
                file.write(str(coord[0]) + ',' + str(coord[1]) + ',' + str(coord[2]) + '\n')
            file.close()


def plot_clustering():
    # This function was made to plot the results of clustering with different height in the camera position
    # height = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    # height = [105, 110, 115, 120, 125]
    height = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125]
    nb_point = []
    trueLength = []
    dpList = []
    for i in range(len(height)):
        trueLength.append(height[i] * math.tan(37.5 * math.pi / 180) * 2)
    for i in range(len(height)):
        nb_point_per_height = []
        dp = trueLength[i] / 640
        dpList.append(dp)
        for j in range(10):
            ncoord = read_data([height[i], (j+1)*5], 'height')
            nb_point_per_height.append(len(ncoord))
            leaves, nb_classes, pred = get_leaves(ncoord, dp)
            ncoord2 = get_numpy_array(leaves)
            # if (height[i] == 50 or height[i] == 125) and j == 9:
            if j == 9 and height[i] % 25 == 0:
                plot_data(ncoord[:-2], ncoord[:-2, 2])
                plot_data(ncoord2, ncoord2[:, 3])
            # if j == 2 or j == 4 or j == 7 or j == 9:
            if j == 4 or j == 9:
                print('debug stop')
            if i % 5 == 0 and j == 9:
                print('pause')
        nb_point.append(nb_point_per_height)


def get_dp_list():
    height = (np.arange(40) + 1) * 5 + 45
    trueLength = []
    dpList = []
    for i in range(len(height)):
        trueLength.append(height[i] * math.tan(37.5 * math.pi / 180) * 2)
    for i in range(len(height)):
        dp = trueLength[i] / 640
        dpList.append(dp)
    temp = np.append(height[np.newaxis].T, np.array(dpList)[np.newaxis].T, axis=1)
    return np.append(temp, np.array(dpList)[np.newaxis].T * 4, axis=1)


def make_a_full_simulation():
    resolution = [1280, 720]
    height = 50
    trueLength = 50

    # Generate the plant
    plant = create_basil()
    plant.initialize(True)
    listPlant = list()
    plantAllTime = {}
    for i in range(1, time):
        plant.simulate(1)
        listPlant.append(plant.copy())
        plantDict = get_real_parameters(plant)
        plantAllTime[i] = plantDict

    # Retrieve real values of phenotype
    fileName = 'path/name_wanted.txt'
    write_real_parameters_file(fileName, plantAllTime)

    success = read_real_parameters_file('Real_parameters_of_plant.txt')
    isASuccess = (success == plantAllTime)  # Just to check on debug mod if there was no error with the file writing

    # Making the ray tracing
    for i in range(len(plantDict)):
        coords = camera_view(plantDict[i], height=height, resolution=resolution, trueLength=trueLength,
                             isPlotted=False, plantType="basil")
        file = open("path/name_wanted_t{}.txt".format(i+1), "w")
        for coord in coords:
            file.write(str(coord[0]) + ',' + str(coord[1]) + ',' + str(coord[2]) + '\n')
        file.close()

    # Retrieve the phenotype by the camera view
    simulation(time)  # Need to change lines 75 and 205 of TimeEvolution to have the correct path of file


def main():
    for_the_presentation()
    make_a_full_simulation()


if __name__ == "__main__":
    main()



