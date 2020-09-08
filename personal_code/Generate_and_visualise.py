# #<---Click this triangle to load python libraries/ download CPlantBox from GitHub/ Make a plant and visualize it
#############loading other python library or packages. Only need to run once at the start.##########
import os
import sys
import vtk
sys.path.append("..")
import plantbox as pb  # CPlantBox Python Binding
# from vtk.util import numpy_support as VN
# import tutorial.examples.python.vtk_plot as vp
# from tutorial.examples.python.vtk_tools import *

sys.path.append("../tutorial")
from jupyter.CPlantBox_PiafMunch import *
from Create_Plants import *
from vtk_plot_bis import *
import math
import matplotlib.pyplot as plt
import numpy as np
from Get_real_values import *
# plotly.__version__


time = 50  # how many days the plant need to grow, make it smaller, for example 15 to see if the plant becomes smaller


# plant = create_basil()
# plant.initialize(True)
# listPlant = list()
# plantAllTime = {}
# for i in range(1, time):
#     plant.simulate(1)
#     listPlant.append(plant.copy())
#     plantDict = get_real_parameters(plant)
#     plantAllTime[i] = plantDict
# fileName = 'Real_parameters_of_plant.txt'
# write_real_parameters_file(fileName, plantAllTime)
#
# success = read_real_parameters_file('Real_parameters_of_plant.txt')
# isASuccess = (success == plantAllTime)
# print('ppp')
# # Rename file that ara wrongly named
# # i = 47
# # while i >= 0:
# #     old = "/home/roberto/Téléchargements/CPlantBox-vierge/ArabidopsisData/arabidopsis_coordinates_t" + str(i) + ".txt"
# #     new = "/home/roberto/Téléchargements/CPlantBox-vierge/ArabidopsisData/arabidopsis_coordinates_t" + str(i+1) + ".txt"
# #     os.rename(old, new)
# #     i -= 1
# # print('yeah')
#
#
# # for dt in range(1, int(time)):
# #     plant1 = CPlantBox(name, dt, name) # make a plant object in python
# #     # Visualization
# #     fig = visual_plant(plant1)
# #     fig.show()
#
# # plant1 = CPlantBox(name, 75, name)  # make a plant object in python
# # # argh = CPlantBox_PiafMunch(name, 50, name)
# # # Visualization
# # fig = visual_plant(plant1)
# # fig.show()
# # # # Congrats! Now you are ready to use the CPlantBox
#
# # for dt in range(1, int(time)):
# #     plant1 = CPlantBox(name, dt*5, name)  # make a plant object in python
# #     fig = visual_plant(plant1)
# #     fig.show()
#
# # plant = create_basil()
# # plant = create_arabidopsis()
# plant = creacrea()
# plant.initialize(True)
# plant.simulate(time)
# fig = visual_plant(plant)
# fig.show()
# # test = plant.getOrganRandomParameter(3)
# # test[0].lmax = 50
# # test2 = plant.getOrganRandomParameter(3)
# plant.simulate(time-5)
# fig = visual_plant(plant)
# fig.show()
# for i in range(5):
#     plant.simulate(1)
#     fig = visual_plant(plant)
#     fig.show()
# test = plant.getOrganRandomParameter(3)
# test[0].lmax = 50
# test2 = plant.getOrganRandomParameter(3)
# plant.simulate(30)
# fig = visual_plant(plant)
# fig.show()
# for i in range(5):
#     # plant.simulate((i+1)/2)
#     plant.simulate(i+1)
#     fig = visual_plant(plant)
#     fig.show()
# print('Finito')
#
# # # If you want to plot using vtk then simulate the camera recording
# # plot_plant(plant, "creationTime", plantType="basil")
# # coords = camera_view(plant, 50, [1280, 720], 50, False)
# # print('debug stop')
# # file = open("basil_coordinates.txt", "w")
# # mot = []
# # for coord in coords:
# #     file.write(str(coord[0]) + ',' + str(coord[1]) + ',' + str(coord[2]) + '\n')
# # file.close()
#
# # # If you want to simulate the camera view on each dt step and then write the result in a txt file
# # for dt in range(1, time):
# #     plant1 = create_basil()
# #     plant1.initialize(True)
# #     plant1.simulate(dt)
# #     coords = vp.camera_view(plant1, 50, [1280, 720], 50, False)
# #     # file = open("BasilData/basil_coordinates_t{}.txt".format(dt), "w")
# #     file = open("Arabidopsis/basil_coordinates_t{}.txt".format(dt), "w")
# #     mot = []
# #     for coord in coords:
# #         file.write(str(coord[0]) + ',' + str(coord[1]) + ',' + str(coord[2]) + '\n')
# #     file.close()
#
# # plant1 = creacrea()
# plant1 = create_basil()
# plant1.initialize(True)
# listPlant = []
# for dt in range(1, time):
#     plant1.simulate(1)  # Simulate one time step further than before
#     listPlant.append(plant1.copy())
#
#
# fig = visual_plant(listPlant[-1])
# fig.show()
#
# print('mate kudasai')
# plot_plant(listPlant[-1], "plant", plantType="basil", stamp=1)
#
# temp_coordinates = python_nodes(listPlant[-1])
# y_min = np.amin(temp_coordinates[:, 4])
# y_max = np.amax(temp_coordinates[:, 4])
# z_min = np.amin(temp_coordinates[:, 5])
# z_max = np.amax(temp_coordinates[:, 5])
#
# for j in range(len(listPlant)):
#     plant = listPlant[j]
#     coordinates = python_nodes(plant)
#     labels = list()
#     for i in range(len(coordinates[:, 1])):
#         if coordinates[i, 1] == 2.0:
#             labels.append("wheat")
#         elif coordinates[i, 1] == 3.0:
#             labels.append("darkgreen")
#         elif coordinates[i, 1] == 4.0:
#             labels.append("lightgreen")
#         else:
#             labels.append("black")
#     plt.scatter(x=coordinates[:, 4], y=coordinates[:, 5], c=labels)
#     plt.ylabel('z axis')
#     plt.xlabel('y axis')
#     plt.title('Growth of basil plant')
#     plt.xlim(y_min, y_max)
#     plt.ylim(z_min, z_max)
#     plt.savefig("/home/roberto/Téléchargements/CPlantBox-vierge/Plot/basil_growth_{}".format(j+1))
#     # fig = visual_plant(plant)
#     # fig.show()
#
# for i in range(len(listPlant)):
#     plot_plant(listPlant[i], "plant", plantType="basil", stamp=i+1)
# print('tchioto')


plant1 = create_basil()
plant1.initialize(True)
listPlant = []
plantAllTime = {}
for dt in range(1, time):
    plant1.simulate(1)  # Simulate one time step further than before
    listPlant.append(plant1.copy())
    plantDict = get_real_parameters(plant1)
    plantAllTime[dt] = plantDict

plot_plant(listPlant[-1], "plant", plantType="basil")

fileName = 'Real_parameters_of_basil2.txt'
write_real_parameters_file(fileName, plantAllTime)

success = read_real_parameters_file(fileName)
isASuccess = (success == plantAllTime)

# resolution = [640, 576]
resolution = [1280, 720]
height = 50
# trueLength = height * math.tan(37.5 * math.pi / 180) * 2  # Multiplied by 2 because we divided the isosceles triangle into 2 rectangle triangle
trueLength = 50

for i in range(len(listPlant)):
    coords = camera_view(listPlant[i], height=height, resolution=resolution, trueLength=trueLength, isPlotted=False, plantType="basil")
    # file = open("/home/roberto/Téléchargements/CPlantBox-vierge/basiltest3/basil_coordinates_t{}.txt".format(i+1), "w")
    # file = open("/home/roberto/Téléchargements/CPlantBox-vierge/ArabidopsisData/arabidopsis_coordinates_t{}.txt".format(i+1), "w")
    file = open("/home/roberto/Téléchargements/CPlantBox-vierge/BasilData2/Basil2_coordinates_t{}.txt".format(i + 1), "w")
    mot = []
    for coord in coords:
        file.write(str(coord[0]) + ',' + str(coord[1]) + ',' + str(coord[2]) + '\n')
    file.close()