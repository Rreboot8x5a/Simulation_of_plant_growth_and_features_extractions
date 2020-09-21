"""
Copyright 2020 Rreboot8x5a

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
The purpose of this script is to retrieve the values of each parameter with the original plant object.
For the error computation, those values will be the exact one.
"""
import plantbox as pb  # CPlantBox Python Binding
from Create_Plants import *
import numpy as np
import math


def get_real_parameters(plant):
    plantChar = {}

    plStem = plant.getPolylines(3)
    if len(plStem) != 0:
        totalHeight = plStem[0][-1].z
    else:
        totalHeight = 0
    plantChar["Height"] = totalHeight

    plLeaf = plant.getPolylines(4)
    organLeafList = plant.getOrgans(4)
    if len(plLeaf) != 0:
        if len(plLeaf) != len(organLeafList):
            print('Error : not the same number of organs with getPolylines and getOrgans !')
        for i in range(len(plLeaf)):
            length = organLeafList[i].getLength() * 0.8
            width = organLeafList[i].getLength() * 0.4  # Needs to be more precised
            # (as it is originally defined as 0.2 * lengthtot, where lengthtot = dist(firstPoint, lastPoint) )

            meanX = 0
            meanY = 0
            meanZ = 0
            for j in range(len(plLeaf[i])):
                meanX += plLeaf[i][j].x
                meanY += plLeaf[i][j].y
                meanZ += plLeaf[i][j].z
            meanX /= len(plLeaf[i])
            meanY /= len(plLeaf[i])
            meanZ /= len(plLeaf[i])
            vector = [meanX - plLeaf[i][0].x, meanY - plLeaf[i][0].y, meanZ - plLeaf[i][0].z]
            horizontal = np.sqrt(vector[0] ** 2 + vector[1] ** 2)
            angle = math.atan(vector[2] / horizontal) * 180 / math.pi
            angle = 90 - angle

            z = plLeaf[i][0].z

            if i < len(plLeaf) - 4:  # For the basil, only the 4 first leaves are not hidden
                isHidden = True
            else:
                isHidden = False

            plantChar[i+1] = {
                "Leaf length": length,
                "Width": width,
                "Angle": angle,
                "Node height": z,
                "Total length": length / 0.8,
            }
    return plantChar


def write_real_parameters_file(name, totalDictionary):
    f = open(name, 'w')
    f.write(str(totalDictionary))
    f.close()


def read_real_parameters_file(name):
    f = open(name, 'r')
    stringDict = f.read()
    allPlantTime = eval(stringDict)
    return allPlantTime


def compute_errors():
    print('just a random sentence')
    # TODO compute the error precisely
