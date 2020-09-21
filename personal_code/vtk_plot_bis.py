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
import sys
sys.path.append("..")
import plantbox as pb
sys.path.append("../tutorial/examples/python")
from vtk_tools import *
from vtk_plot import *

import numpy as np
import vtk
import matplotlib.pyplot as plt
import math
from scipy.optimize import fsolve
from sympy import Symbol, nsolve
import random as rand
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

""" 
A code mainly based on the script VTK_plot from CPlantBox.
The purpose of this script is to be able to plot plant using VTK by adding leaves on the model.
Moreover, it simulates the nadir camera view and retrieve the coordinates of the viewed points. 
"""


def plot_plant(pd, p_name, win_title="", render=True, plantType="basil"):
    """ plots the plant system (basically just a copy of plot_root adapted so it can add leaves)
    @param pd         the polydata representing the plant system (lines, or polylines)
    @param p_name     parameter name of the data to be visualized
    @param win_title  the windows titles (optionally, defaults to p_name)
    @param render     render in a new interactive window (default = True)
    @param plantType  define the plant type (basil or arabidopsis), arabidopsis model use some tricks to work
    @return a tuple of a vtkActor and the corresponding color bar vtkScalarBarActor
    """
    leafLines = pd.getPolylines(4)  # The plant is constitute of multiple lines, the parameter "4" allow to retrieve
    # only the lines coming from leaf organs
    if plantType == "arabidopsis":
        rosetteLines = pd.getPolylines(3)  # If the arabidopsis is generated, we need the stem lines because one subType
        # of stem is a trick to create more leaves
        organs = pd.getOrgans(3)  # Create a list of organs of the same size of rosetteLines, used to retrieve subType

    if isinstance(pd, pb.RootSystem):
        pd = segs_to_polydata(pd, 1.)

    if isinstance(pd, pb.SegmentAnalyser):
        pd = segs_to_polydata(pd, 1.)

    if isinstance(pd, pb.Plant):
        pd = segs_to_polydata(pd, 1.)

    if win_title == "":
        win_title = p_name

    pd.GetPointData().SetActiveScalars("radius")  # for the the filter
    tubeFilter = vtk.vtkTubeFilter()
    tubeFilter.SetInputData(pd)
    tubeFilter.SetNumberOfSides(9)
    tubeFilter.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
    tubeFilter.SetCapping(True)
    tubeFilter.Update()

    leaves = vtk.vtkPolyData()
    # leaves = create_leaf(leafLines)
    leaves = create_leaves(leafLines)

    appendFilter = vtk.vtkAppendPolyData()  # The append filter allows to unite multiple sets of polygonal data
    appendFilter.AddInputData(leaves)  # The leaves are added to the append filter
    if plantType == "arabidopsis":
        rosette = create_rosette(rosetteLines, organs)
        appendFilter.AddInputData(rosette)
    appendFilter.AddInputData(tubeFilter.GetOutput())  # The stems and the roots are added to the append filter
    appendFilter.Update()

    # Just a copy paste from the orginal VTK_plot script
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(appendFilter.GetOutputPort())
    mapper.Update()
    mapper.ScalarVisibilityOn()
    mapper.SetScalarModeToUseCellFieldData()  # maybe because radius is active scalar in point data?
    mapper.SetArrayName(p_name)
    mapper.SelectColorArray(p_name)
    mapper.UseLookupTableScalarRangeOn()
    plantActor = vtk.vtkActor()
    plantActor.SetMapper(mapper)

    listActor = list()
    listActor.append(plantActor)  # You can add another actor to the list if you want another data to be plotted

    lut = create_lookup_table()  # 24
    scalar_bar = create_scalar_bar(lut, pd, p_name)  # vtkScalarBarActor
    mapper.SetLookupTable(lut)

    if render:
        render_window(listActor, win_title, scalar_bar, pd.GetBounds()).Start()
    return plantActor, scalar_bar


def intersection(surfaces, ray):
    # Surfaces is as VTKmodifiedBSPtree object
    # Ray contains the coordinates of 2 points defining the line

    # Mainly based on "https://lorensen.github.io/VTKExamples/site/Python/GeometricObjects/PolygonIntersection/"
    # tutorial, adapted because ModifiedBSPTree is used instead of a set of polygons.
    p1 = ray[0]  # The coordinates of the light source of the kinect
    p2 = ray[1]  # The coordinates of the intersection between the ground and the ray
    tolerance = 0.001
    bestT = 2  # the best t parameter for the intersection, 1 is the worst possible value, 0 is the best one
    bestID = -300000
    bestX = [-100, -100, -100]
    atLeastOne = False
    t = vtk.mutable(0)  # Parametric coordinate of intersection (0 (corresponding to p1) to 1 (corresponding to p2))
    x = [-100, -100, -100]  # Setting an impossible value for the intersection (just to check problematic case)
    pcoords = [0.0, 0.0, 0.0]
    subId = vtk.mutable(0)
    iD = surfaces.IntersectWithLine(p1, p2, tolerance, t, x, pcoords, subId)

    # This part was initially there because "surfaces" parameter was a list of polygons (so a ray can intersect
    # with several polygons and we need to find the nearest intersection with the light source)
    # TODO refactor the code so it benefits more of the specification of modifiedBSPtree (see comment above)
    if iD == 1:
        atLeastOne = True
        if t < bestT:
            # bestID = i
            bestT = t
            bestX = x
    return bestID, bestT, bestX, atLeastOne


def get_rays(length, width, height, trueLength):
    # The function only return the intersection between the rays and the ground (xy plan)
    # The lines are defined by 2 points
    # The camera is set in the position (0, 0, z),
    # it is not restrictive as we can move easily the position of the seed of the plant
    # Truelength is the distance between two extreme pixels if they both hit the ground
    # It would have been better to  use the open angle of the light source (it would have been more understandable)
    rays = []
    centerOfEverything = [0, 0, height]
    trueWidth = trueLength/length * width
    dp = trueLength/length  # the length of a pixel if it hits the ground
    for i in range(length):
        posx = i - length/2
        posx += 0.5
        posx *= dp  # Position x of the center of the pixel if it hits the grounds
        for j in range(width):
            posy = j - width / 2
            posy += 0.5
            posy *= dp  # # Position y of the center of the pixel if it hits the grounds
            rays.append([posx, posy, 0])
    return rays


def camera_view(pd, height, resolution, trueLength, isPlotted, plantType="basil"):
    # Simulate the nadir camera view and retrieve the coordinates of the viewed points.
    # Based on plot_plant (the plotting part is removed and the ray tracing part is added)
    nods = np.array([np.array(s) for s in pd.getNodes()])
    leafLines = pd.getPolylines(4)

    if plantType == "arabidopsis":
        rosetteLines = pd.getPolylines(3)  # If the arabidopsis is generated, we need the stem lines because one subType
        # of stem is a trick to create more leaves
        organs = pd.getOrgans(3)  # Create a list of organs of the same size of rosetteLines, used to retrieve subType


    if isinstance(pd, pb.RootSystem):
        pd = segs_to_polydata(pd, 1.)

    if isinstance(pd, pb.SegmentAnalyser):
        pd = segs_to_polydata(pd, 1.)

    if isinstance(pd, pb.Plant):
        pd = segs_to_polydata(pd, 1.)

    pd.GetPointData().SetActiveScalars("radius")  # for the the filter
    tubeFilter = vtk.vtkTubeFilter()
    tubeFilter.SetInputData(pd)
    tubeFilter.SetNumberOfSides(9)
    tubeFilter.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
    tubeFilter.SetCapping(True)
    tubeFilter.Update()
    stemModel = tubeFilter.GetOutput()

    # leaves = create_leaf(leafLines)
    leaves = create_leaves(leafLines)
    appendFilter = vtk.vtkAppendPolyData()  # So we can merge the roots, the stems and the leaves together
    appendFilter.AddInputData(leaves)
    appendFilter.AddInputData(stemModel)
    if plantType == "arabidopsis":
        rosette = create_rosette(rosetteLines, organs)
        appendFilter.AddInputData(rosette)

    appendFilter.Update()

    plantWithLeaves = appendFilter.GetOutput()

    bsp = vtk.vtkModifiedBSPTree()
    bsp.SetDataSet(plantWithLeaves)
    bsp.BuildLocator()

    rays = get_rays(resolution[0], resolution[1], height, trueLength)
    position = []
    for i in range(len(rays)):
        if i % 100 == 0:
            print(str(i))
        bestID, bestT, bestX, ok = intersection(bsp, [[0, 0, height], rays[i]])
        if ok == True:
            position.append(bestX)

    rays = np.array(rays)
    position.append([0, 0, height])
    position = np.array(position)
    nods = np.append(nods, np.zeros((len(nods), 1)), axis=1)
    position = np.append(position, np.ones((len(position), 1)), axis=1)
    final = np.append(position, nods, axis=0)

    # Just a part of the code used for checking result (probably useless now)
    if isPlotted:
        x = position[:, 0]
        y = position[:, 1]
        z = position[:, 2]
        subfig = go.Scatter3d(
            x=final[:, 0],
            y=final[:, 1],
            z=final[:, 2],
            # x=position[:200000, 0],
            # y=position[:200000, 1],
            # z=position[:200000, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=final[:, 3],  # nodes_cor.T[1] is organ type, nodes_cor.T[2] is the connection number of a node
                colorscale='Viridis',
                opacity=0.8
            )
        )
        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{'type': 'surface'}]])
        fig.add_trace(subfig)
        fig.update_layout(scene_aspectmode='data', )
        fig.show()

    return position[:, :-1]


def create_leaves(leafLines):
    appendFilter = vtk.vtkAppendPolyData()
    for i in range(len(leafLines)):  # For each leaf
        lengthtotx = -(leafLines[i][0].x - leafLines[i][-1].x)
        lengthtoty = -(leafLines[i][0].y - leafLines[i][-1].y)
        lengthtotz = -(leafLines[i][0].z - leafLines[i][-1].z)
        lengthtot = lengthtotx ** 2 + lengthtoty ** 2 + lengthtotz ** 2

        # The distance between the first node of the leaf and the last node of the leaf
        lengthtot = math.sqrt(lengthtot)  # Norm in numpy could have done the job
        x = 0
        for j in range(len(leafLines[i])-1):  # For each segment line composing the leaf
            if j >= len(leafLines[i])*0.2:  # Just to have a petiole
                lengthx = -(leafLines[i][j].x - leafLines[i][j+1].x)
                lengthy = -(leafLines[i][j].y - leafLines[i][j+1].y)
                lengthz = -(leafLines[i][j].z - leafLines[i][j+1].z)
                length = lengthx ** 2 + lengthy ** 2 + lengthz ** 2
                length = math.sqrt(length)  # The length of the segment
                # # Old implementation with rectangle leaves
                # p0 = [-lengthtot / 4, 0, 0]
                # p1 = [-lengthtot / 4, length, 0]
                # p2 = [lengthtot / 4, length, 0]
                # p3 = [lengthtot / 4, 0, 0]
                nb_segments = int(len(leafLines[i])*0.8)
                x2 = (x/(nb_segments/2 - 0.5)) - 1
                y1 = (-x2**2 + 1) * lengthtot * 0.2
                # If we want something else than rectangle, we define a function f(x) = x**2
                y2 = (-(x2 + 1/(nb_segments / 2 - 0.5))**2 + 1) * lengthtot * 0.2  # there we take x2 + dx
                x += 1

                # We define a symmetric trapezoid at the origin
                p0 = [-y1, 0, 0]
                p1 = [-y2, length, 0]
                p2 = [y2, length, 0]
                p3 = [y1, 0, 0]

                points = vtk.vtkPoints()
                points.InsertNextPoint(p0)
                points.InsertNextPoint(p1)
                points.InsertNextPoint(p2)
                points.InsertNextPoint(p3)

                # Create the polygon
                polygon = vtk.vtkPolygon()
                polygon.GetPointIds().SetNumberOfIds(4)  # make a quad
                polygon.GetPointIds().SetId(0, 0)
                polygon.GetPointIds().SetId(1, 1)
                polygon.GetPointIds().SetId(2, 2)
                polygon.GetPointIds().SetId(3, 3)

                # Add the polygon to a list of polygons
                polygons = vtk.vtkCellArray()
                polygons.InsertNextCell(polygon)

                # Create a PolyData
                polygonPolyData = vtk.vtkPolyData()
                polygonPolyData.SetPoints(points)
                polygonPolyData.SetPolys(polygons)

                transfo = vtk.vtkTransform()
                thetarad = math.atan(lengthz / (math.sqrt(lengthy ** 2 + lengthx ** 2)))
                theta = thetarad * 360 / (2 * math.pi)

                phi = math.atan(lengthtoty / lengthtotx)
                phi = phi * 360 / (2 * math.pi)

                isOpposate = 0
                if lengthx < 0:
                    isOpposate = 180  # atan is set between -pi/2 and pi/2 so I add 180° if the angle is not in this interval

                # Moving the polygon to the segment line of the leaf
                transfo.Translate([leafLines[i][j].x, leafLines[i][j].y, leafLines[i][j].z])
                transfo.RotateWXYZ(theta, [1, 0, 0])
                transfo.RotateWXYZ(phi - 90 + isOpposate,
                                   [0, math.sin(thetarad), math.cos(thetarad)])  # phi-90 because the leaf was oriented
                # toward y. the axes are bind to the polygon, so the second rotation does not use z absolute axe

                transformFilter = vtk.vtkTransformPolyDataFilter()
                transformFilter.SetTransform(transfo)
                transformFilter.Update()

                transformFilter.SetInputData(polygonPolyData)
                transformFilter.Update()

                appendFilter.AddInputData(transformFilter.GetOutput())

    appendFilter.Update()
    leaves = appendFilter.GetOutput()

    return leaves


def create_rosette(stemLines, organs):
    # Same behavior as create_leaves, this function is used because of the trick for the rosette
    appendFilter = vtk.vtkAppendPolyData()
    counter = 0
    for i in range(len(stemLines)):
        # if stemLines[i][0].z == -3 and np.abs(stemLines[i][-1].x) + np.abs(stemLines[i][-1].y) > 0.1:
        if organs[i].getParameter('subType') == 4:
            # Only keeps the tricky stem that are the rosette leaves.
            lengthtotx = -(stemLines[i][0].x - stemLines[i][-1].x)
            lengthtoty = -(stemLines[i][0].y - stemLines[i][-1].y)
            lengthtotz = -(stemLines[i][0].z - stemLines[i][-1].z)
            lengthtot = lengthtotx ** 2 + lengthtoty ** 2 + lengthtotz ** 2
            lengthtot = math.sqrt(lengthtot)

            x = 0
            for j in range(len(stemLines[i]) - 1):
                if j >= len(stemLines[i]) * 0.2:
                    lengthx = -(stemLines[i][j].x - stemLines[i][j + 1].x)
                    lengthy = -(stemLines[i][j].y - stemLines[i][j + 1].y)
                    lengthz = -(stemLines[i][j].z - stemLines[i][j + 1].z)
                    length = lengthx ** 2 + lengthy ** 2 + lengthz ** 2
                    length = math.sqrt(length)
                    nb_segments = int(len(stemLines[i]) * 0.8)
                    x2 = (x / (nb_segments / 2 - 0.5)) - 1
                    y1 = (-x2 ** 2 + 1) * lengthtot * 0.2
                    y1 = (-x ** 3 + x ** 2) * lengthtot
                    # If we want something else than rectangle, we define a function f(x) = x**2
                    y2 = (-(x2 + 1 / (nb_segments / 2 - 0.5)) ** 2 + 1) * lengthtot * 0.2  # there we take x2 + dx
                    y2 = (-(x + 1/nb_segments) ** 3 + (x + 1/nb_segments) ** 2) * lengthtot
                    x += 1/nb_segments
                    p0 = [-y1, 0, 0]
                    p1 = [-y2, length, 0]
                    p2 = [y2, length, 0]
                    p3 = [y1, 0, 0]
                    points = vtk.vtkPoints()
                    points.InsertNextPoint(p0)
                    points.InsertNextPoint(p1)
                    points.InsertNextPoint(p2)
                    points.InsertNextPoint(p3)

                    # Create the polygon
                    polygon = vtk.vtkPolygon()
                    polygon.GetPointIds().SetNumberOfIds(4)  # make a quad
                    polygon.GetPointIds().SetId(0, 0)
                    polygon.GetPointIds().SetId(1, 1)
                    polygon.GetPointIds().SetId(2, 2)
                    polygon.GetPointIds().SetId(3, 3)

                    # Add the polygon to a list of polygons
                    polygons = vtk.vtkCellArray()
                    polygons.InsertNextCell(polygon)

                    # Create a PolyData
                    polygonPolyData = vtk.vtkPolyData()
                    polygonPolyData.SetPoints(points)
                    polygonPolyData.SetPolys(polygons)

                    transfo = vtk.vtkTransform()
                    thetarad = math.atan(lengthz / (math.sqrt(lengthy ** 2 + lengthx ** 2)))
                    theta = thetarad * 360 / (2 * math.pi)

                    phi = math.atan(lengthtoty / lengthtotx)
                    phi = phi * 360 / (2 * math.pi)

                    isOpposate = 0
                    if lengthx < 0:
                        isOpposate = 180  # atan is set between -pi/2 and pi/2 so I add 180° if the angle is not in this interval

                    transfo.Translate([stemLines[i][j].x, stemLines[i][j].y, stemLines[i][j].z])
                    transfo.RotateWXYZ(theta, [1, 0, 0])
                    transfo.RotateWXYZ(phi - 90 + isOpposate,
                                       [0, math.sin(thetarad),
                                        math.cos(thetarad)])  # phi-90 because the leaf was oriented
                    # toward y. the axes are bind to the polygon, so the second rotation does not use z absolute axe

                    transformFilter = vtk.vtkTransformPolyDataFilter()
                    transformFilter.SetTransform(transfo)
                    transformFilter.Update()

                    transformFilter.SetInputData(polygonPolyData)
                    transformFilter.Update()

                    appendFilter.AddInputData(transformFilter.GetOutput())

    appendFilter.Update()
    leaves = appendFilter.GetOutput()

    return leaves

