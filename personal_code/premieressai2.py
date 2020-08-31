"""
Old version of Generate_and_visualise
Need to be removed at the end of the project
"""

# #<---Click this triangle to load python libraries/ download CPlantBox from GitHub/ Make a plant and visualize it
#############loading other python library or packages. Only need to run once at the start.##########
import os
import sys
import vtk
# from vtk.util import numpy_support as VN
import tutorial.examples.python.vtk_plot as vp
from tutorial.examples.python.vtk_tools import *
from sympy import Symbol, nsolve

### Example5a est cool car il rajoute l'épaisseur

# Loading specific python scripts for CPlantBox and CRootBox
os.chdir("/home/roberto/CplantBox/CPlantBox-master/tutorial/jupyter")
os.chdir("/home/roberto/CplantBox/CPlantBox-master")
from tutorial.jupyter.CPlantBox_PiafMunch import *
# plotly.__version__

####################### first we create a Heliantus plant##########################################
name = "0.xml" # parameter name
# name = "fuck.xml" # parameter name
# name = "leaf_opposite_decussate.xml"
# name = "monopodial2.xml"
# name = "sympodial_dichasium.xml"
# name = "Anagallis_femina_Leitner_2010.xml"

# here are some optional parameter files to be tested
# name = "PMA2018" # Simulate a small plant with 3 leaves and two lateral root, you can comment the heliantus line and uncomment this line to see what happend.
time = 51  # how many days the plant need to grow, make it smaller, for example 15 to see if the plant becomes smaller

# for dt in range(1, int(time)):
#     plant1 = CPlantBox(name, dt, name) # make a plant object in python
#     # Visualization
#     fig = visual_plant(plant1)
#     fig.show()

plant1 = CPlantBox(name, 50, name)  # make a plant object in python
# # Visualization
# fig = visual_plant(plant1)
# fig.show()
vp.plot_roots(plant1, "creationTime")
# coords = vp.camera_view(plant1, 50, [1280, 720], 50, False)
# file = open("basil_coordinates.txt", "w")
# mot = []
# for coord in coords:
#     file.write(str(coord[0]) + ',' + str(coord[1]) + ',' + str(coord[2]) + '\n')
# file.close()

#open and read the file after the appending:
# f = open("demofile2.txt", "r")
# print(f.read())

print('fin')
#fig = visual_plant(plant1)
#fig.show()

# for dt in range(1, int(time)):
#     plant1 = CPlantBox(name, dt*5, name) # make a plant object in python
#     vp.plot_roots(plant1, "creationTime")

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
