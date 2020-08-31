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
# plotly.__version__


time = 50  # how many days the plant need to grow, make it smaller, for example 15 to see if the plant becomes smaller

# for dt in range(1, int(time)):
#     plant1 = CPlantBox(name, dt, name) # make a plant object in python
#     # Visualization
#     fig = visual_plant(plant1)
#     fig.show()

# plant1 = CPlantBox(name, 75, name)  # make a plant object in python
# # argh = CPlantBox_PiafMunch(name, 50, name)
# # Visualization
# fig = visual_plant(plant1)
# fig.show()
# # # Congrats! Now you are ready to use the CPlantBox

# for dt in range(1, int(time)):
#     plant1 = CPlantBox(name, dt*5, name)  # make a plant object in python
#     fig = visual_plant(plant1)
#     fig.show()
    
    
# plant = create_basil()
# plant = create_arabidopsis()
plant = creacrea()
plant.initialize(True)
plant.simulate(time)
fig = visual_plant(plant)
fig.show()

# # If you want to plot using vtk then simulate the camera recording
plot_plant(plant, "creationTime", plantType="arabidopsis")
coords = camera_view(plant, 50, [1280, 720], 50, False)
print('debug stop')
# file = open("basil_coordinates.txt", "w")
# mot = []
# for coord in coords:
#     file.write(str(coord[0]) + ',' + str(coord[1]) + ',' + str(coord[2]) + '\n')
# file.close()

# # If you want to simulate the camera view on each dt step and then write the result in a txt file
# for dt in range(1, time):
#     plant1 = create_basil()
#     plant1.initialize(True)
#     plant1.simulate(dt)
#     coords = vp.camera_view(plant1, 50, [1280, 720], 50, False)
#     file = open("BasilData/basil_coordinates_t{}.txt".format(dt), "w")
#     mot = []
#     for coord in coords:
#         file.write(str(coord[0]) + ',' + str(coord[1]) + ',' + str(coord[2]) + '\n')
#     file.close()
