import sys
sys.path.append("..")
import plantbox as pb
sys.path.append("../tutorial/examples/python")
from vtk_tools import *
from vtk_plot import *

import time
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
VTK Plot, by Daniel Leitner (refurbished 06/2020) 

to make interactive vtk plot of root systems and soil grids
"""


def plot_plant(pd, p_name, win_title = "", render = True):
    """ plots the plant system 
    @param pd         the polydata representing the plant system (lines, or polylines)
    @param p_name     parameter name of the data to be visualized
    @param win_title  the windows titles (optionally, defaults to p_name)
    @param render     render in a new interactive window (default = True)
    @return a tuple of a vtkActor and the corresponding color bar vtkScalarBarActor
    """
    poly = pd.getPolylines()
    utile = pd.getPolylines(4)
    if True:
        utile2 = pd.getPolylines(3)
    organ = pd.getOrgans()
    organ2 = pd.getOrgans(4)
    # param = organ2[1].getParameter('length')
    identity = organ2[1].getId()
    nods = np.array([np.array(s) for s in pd.getNodes()])
    # feuille = []
    #
    # for i in range(len(organ)):
    #     if organ[i].getParameter('type') == 4:  # Only doing it if the organ is a leaf
    #         for j in range(len(poly[i])-1):
    #             reps = []
    #             p1 = poly[i][j]
    #             p2 = poly[i][j+1]
    #             # v1 = [0, 0, 0]
    #             # v1[0] = p2.x - p1.x
    #             # v1[1] = p2.y - p1.y
    #             # v1[2] = p2.z - p1.z
    #             # x = Symbol('x')
    #             # y = Symbol('y')
    #             # z = Symbol('z')
    #             # rep = nsolve((x**2 + y ** 2 - 4, (x - p2.x) * v1[0] + (y - p2.y) * v1[1], z - p2.z), (x, y, z), (p2.x, p2.y, p2.z))
    #             x1 = rand.uniform(p1.x, p2.x)
    #             y1 = rand.uniform(p1.y, p2.y)
    #             z1 = rand.uniform(p1.z, p2.z)
    #             reps.append([x1, y1, z1])
    #         feuille.append(reps)

    if isinstance(pd, pb.RootSystem):
        pd = segs_to_polydata(pd, 1.)

    if isinstance(pd, pb.SegmentAnalyser):
        pd = segs_to_polydata(pd, 1.)

    if isinstance(pd, pb.Plant):
        pd = segs_to_polydata(pd, 1.)

    if win_title == "":
        win_title = p_name

    lines = pd.GetLines()
    pd.GetPointData().SetActiveScalars("radius")  # for the the filter
    tubeFilter = vtk.vtkTubeFilter()
    tubeFilter.SetInputData(pd)
    tubeFilter.SetNumberOfSides(9)
    tubeFilter.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
    tubeFilter.SetCapping(True)
    tubeFilter.Update()

    print('réussi!')

    # zizi est un vtk object, mais quoi ?
    # bubu = list()
    # # blob = zizi.GetNumberOfCells()
    # for i in range(zizi.GetNumberOfCells()):
    #     bubu.append(zizi.GetCellType(i))
    # pipi = zizi.GetPolys()
    # baba = list()
    # for i in range(len(bubu)):
    #     if bubu[i] == 5:
    #         baba.append(i)

    print('Commentaires pour créer des STL, à nettoyer en fin de projet si inutile.')

    # filename = "test.stl"
    #
    # # Write the stl file to disk
    # stlWriter = vtk.vtkSTLWriter()
    # stlWriter.SetFileName(filename)
    # stlWriter.SetInputConnection(tubeFilter.GetOutputPort())
    # stlWriter.Write()
    #
    # # Read and display for verification
    # reader = vtk.vtkSTLReader()
    # reader.SetFileName(filename)
    #
    # mapper = vtk.vtkPolyDataMapper()
    # if vtk.VTK_MAJOR_VERSION <= 5:
    #     mapper.SetInput(reader.GetOutput())
    # else:
    #     mapper.SetInputConnection(reader.GetOutputPort())
    #
    # actor = vtk.vtkActor()
    # actor.SetMapper(mapper)
    #
    # # Create a rendering window and renderer
    # ren = vtk.vtkRenderer()
    # renWin = vtk.vtkRenderWindow()
    # renWin.AddRenderer(ren)
    #
    # # Create a renderwindowinteractor
    # iren = vtk.vtkRenderWindowInteractor()
    # iren.SetRenderWindow(renWin)
    #
    # # Assign actor to the renderer
    # ren.AddActor(actor)
    #
    # # Enable user interface interactor
    # iren.Initialize()
    # renWin.Render()
    # iren.Start()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(tubeFilter.GetOutputPort())
    mapper.Update()
    mapper.ScalarVisibilityOn()
    mapper.SetScalarModeToUseCellFieldData()  # maybe because radius is active scalar in point data?
    mapper.SetArrayName(p_name)
    mapper.SelectColorArray(p_name)
    mapper.UseLookupTableScalarRangeOn()
    plantActor = vtk.vtkActor()
    plantActor.SetMapper(mapper)

    popo = vtk.vtkPolyData()
    # popo = create_leaf(utile)
    popo = create_basil_leaf(utile)

    appendFilter = vtk.vtkAppendPolyData()
    appendFilter.AddInputData(popo)
    if True:  # This is teh arabidopsis
        rosette = vtk.vtkPolyData()
        rosette = create_rosette(utile2)
        appendFilter.AddInputData(rosette)
    appendFilter.AddInputData(tubeFilter.GetOutput())
    appendFilter.Update()
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

    quequecest = appendFilter.GetOutput()

    # transformFilter.GetOutput()
    # mapper2 = vtk.vtkPolyDataMapper()
    # mapper2.SetInputConnection(transformFilter.GetOutputPort())
    # mapper2.Update()
    # mapper2.ScalarVisibilityOn()
    # mapper2.SetScalarModeToUseCellFieldData()  # maybe because radius is active scalar in point data?
    # mapper2.SetArrayName(p_name)
    # mapper2.SelectColorArray(p_name)
    # mapper2.UseLookupTableScalarRangeOn()
    # yolo = vtk.vtkActor()
    # yolo.SetMapper(mapper2)

    p0 = [0, -1, 0]
    p1 = [10, -1, 0]
    p2 = [10, 1, 0]
    p3 = [0, 1, 0]
    points2 = vtk.vtkPoints()
    points2.InsertNextPoint(p0)
    points2.InsertNextPoint(p1)
    points2.InsertNextPoint(p2)
    points2.InsertNextPoint(p3)

    # Create the polygon
    polygon2 = vtk.vtkPolygon()
    polygon2.GetPointIds().SetNumberOfIds(4)  # make a quad
    polygon2.GetPointIds().SetId(0, 0)
    polygon2.GetPointIds().SetId(1, 1)
    polygon2.GetPointIds().SetId(2, 2)
    polygon2.GetPointIds().SetId(3, 3)

    # Add the polygon to a list of polygons
    polygons2 = vtk.vtkCellArray()
    polygons2.InsertNextCell(polygon2)

    # Create a PolyData
    polygonPolyData2 = vtk.vtkPolyData()
    polygonPolyData2.SetPoints(points2)
    polygonPolyData2.SetPolys(polygons2)

    # transformFilter.GetOutput()

    mapper3 = vtk.vtkPolyDataMapper()
    mapper3.SetInputData(polygonPolyData2)

    papa = vtk.vtkActor()
    papa.SetMapper(mapper3)

    listActor = list()
    listActor.append(plantActor)
    # listActor.append(yolo)
    # listActor.append(papa)
    # listActor.append(surfActor)
    # listActor.append(actor)
    # listActor.append(bspTreeActor)

    lut = create_lookup_table()  # 24
    scalar_bar = create_scalar_bar(lut, pd, p_name)  # vtkScalarBarActor
    mapper.SetLookupTable(lut)

    if render:
        # render_window(plantActor, win_title, scalar_bar, pd.GetBounds()).Start()
        render_window(listActor, win_title, scalar_bar, pd.GetBounds()).Start()
    return plantActor, scalar_bar


def intersection(surfaces, ray):
    p1 = ray[0]
    p2 = ray[1]
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
    if iD == 1:
        atLeastOne = True
        if t < bestT:
            # bestID = i
            bestT = t
            bestX = x

    return bestID, bestT, bestX, atLeastOne


def get_rays(length, width, height, trueLength):
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


def create_leaf(utile):
    appendFilter = vtk.vtkAppendPolyData()
    for i in range(len(utile)):
    # for i in range(1):
        lengthx = -(utile[i][0].x - utile[i][-1].x)
        lengthy = -(utile[i][0].y - utile[i][-1].y)
        lengthz = -(utile[i][0].z - utile[i][-1].z)
        length = lengthx ** 2 + lengthy ** 2 + lengthz ** 2
        length = math.sqrt(length)
        p0 = [-length / 4, 0, 0]
        p1 = [-length / 4, length * 0.8, 0]
        p2 = [length / 4, length * 0.8, 0]
        p3 = [length / 4, 0, 0]
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

        phi = math.atan(lengthy / lengthx)
        phi = phi * 360 / (2 * math.pi)

        isOpposate = 0
        if lengthx < 0:
            isOpposate = 180  # atan is set between -pi/2 and pi/2 so I add 180° if the angle is not in this interval

        basil = False
        if basil:
            transfo.Translate([0, 0, utile[i][0].z])
        else:
            transfo.Translate([utile[i][0].x, utile[i][0].y, utile[i][0].z])
        transfo.RotateWXYZ(theta, [1, 0, 0])
        transfo.RotateWXYZ(phi - 90 + isOpposate, [0, math.sin(thetarad), math.cos(thetarad)])  # phi+90 because the leaf was
        # oriented toward y. the axes are bind to the polygon, so the second rotation does not use z absolute axe
        if True:
            transfo.Translate([0, length*0.2, 0])

        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetTransform(transfo)
        transformFilter.Update()
        # transformFilter.SetTransform(rotate2)
        # transformFilter.SetTransform(translate)

        # transformFilter.SetInputConnection(polygonPolyData.GetOutputPort())
        transformFilter.SetInputData(polygonPolyData)
        transformFilter.Update()

        appendFilter.AddInputData(transformFilter.GetOutput())

    appendFilter.Update()
    leaves = appendFilter.GetOutput()

    return leaves


def create_basil_leaf(utile):
    appendFilter = vtk.vtkAppendPolyData()
    for i in range(len(utile)):
        lengthtotx = -(utile[i][0].x - utile[i][-1].x)
        lengthtoty = -(utile[i][0].y - utile[i][-1].y)
        lengthtotz = -(utile[i][0].z - utile[i][-1].z)
        lengthtot = lengthtotx ** 2 + lengthtoty ** 2 + lengthtotz ** 2
        lengthtot = math.sqrt(lengthtot)
        x = 0
        for j in range(len(utile[i])-1):
            # if True:
            if j >= len(utile[i])*0.2:
                lengthx = -(utile[i][j].x - utile[i][j+1].x)
                lengthy = -(utile[i][j].y - utile[i][j+1].y)
                lengthz = -(utile[i][j].z - utile[i][j+1].z)
                length = lengthx ** 2 + lengthy ** 2 + lengthz ** 2
                length = math.sqrt(length)
                # p0 = [-lengthtot / 4, 0, 0]
                # p1 = [-lengthtot / 4, length, 0]
                # p2 = [lengthtot / 4, length, 0]
                # p3 = [lengthtot / 4, 0, 0]
                nb_segments = int(len(utile[i])*0.8)
                x2 = (x/(nb_segments/2 - 0.5)) - 1
                y1 = (-x2**2 + 1) * lengthtot * 0.2
                # If we want something else than rectangle, we define a function f(x) = x**2
                y2 = (-(x2 + 1/(nb_segments / 2 - 0.5))**2 + 1) * lengthtot * 0.2  # there we take x2 + dx
                x += 1
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

                transfo.Translate([utile[i][j].x, utile[i][j].y, utile[i][j].z])
                transfo.RotateWXYZ(theta, [1, 0, 0])
                transfo.RotateWXYZ(phi - 90 + isOpposate,
                                   [0, math.sin(thetarad), math.cos(thetarad)])  # phi+90 because the leaf was
                # oriented toward y. the axes are bind to the polygon, so the second rotation does not use z absolute axe
                # transfo.Translate([0, length * 0.2, 0])

                transformFilter = vtk.vtkTransformPolyDataFilter()
                transformFilter.SetTransform(transfo)
                transformFilter.Update()
                # transformFilter.SetTransform(rotate2)
                # transformFilter.SetTransform(translate)

                # transformFilter.SetInputConnection(polygonPolyData.GetOutputPort())
                transformFilter.SetInputData(polygonPolyData)
                transformFilter.Update()

                appendFilter.AddInputData(transformFilter.GetOutput())

    appendFilter.Update()
    leaves = appendFilter.GetOutput()

    return leaves


def create_rosette(utile2):
    appendFilter = vtk.vtkAppendPolyData()
    compteur = 0
    for i in range(len(utile2)):
        # for i in range(1):
        if utile2[i][0].z == -3 and np.abs(utile2[i][-1].x) + np.abs(utile2[i][-1].y) > 0.1:
            compteur += 1
            lengthx = -(utile2[i][0].x - utile2[i][-1].x)
            lengthy = -(utile2[i][0].y - utile2[i][-1].y)
            lengthz = -(utile2[i][0].z - utile2[i][-1].z)
            length = lengthx ** 2 + lengthy ** 2 + lengthz ** 2
            length = math.sqrt(length)
            p0 = [0, 0, 0]
            p1 = [-length / 10, length, 0]
            p2 = [length / 10, length, 0]
            # p3 = [length / 4, 0, 0]
            points = vtk.vtkPoints()
            points.InsertNextPoint(p0)
            points.InsertNextPoint(p1)
            points.InsertNextPoint(p2)
            # points.InsertNextPoint(p3)

            # Create the polygon
            polygon = vtk.vtkPolygon()
            polygon.GetPointIds().SetNumberOfIds(3)  # make a quad
            polygon.GetPointIds().SetId(0, 0)
            polygon.GetPointIds().SetId(1, 1)
            polygon.GetPointIds().SetId(2, 2)
            # polygon.GetPointIds().SetId(3, 3)

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

            phi = math.atan(lengthy / lengthx)
            phi = phi * 360 / (2 * math.pi)

            isOpposate = 0
            if lengthx < 0:
                isOpposate = 180  # atan is set between -pi/2 and pi/2 so I add 180° if the angle is not in this interval

            transfo.Translate([0, 0, utile2[i][0].z])
            transfo.RotateWXYZ(theta, [1, 0, 0])
            transfo.RotateWXYZ(phi - 90 + isOpposate,
                               [0, math.sin(thetarad), math.cos(thetarad)])  # phi+90 because the leaf was
            # oriented toward y. the axes are bind to the polygon, so the second rotation does not use z absolute axe
            transfo.Translate([0, length * 0.2, 0])

            transformFilter = vtk.vtkTransformPolyDataFilter()
            transformFilter.SetTransform(transfo)
            transformFilter.Update()
            # transformFilter.SetTransform(rotate2)
            # transformFilter.SetTransform(translate)

            # transformFilter.SetInputConnection(polygonPolyData.GetOutputPort())
            transformFilter.SetInputData(polygonPolyData)
            transformFilter.Update()

            appendFilter.AddInputData(transformFilter.GetOutput())

    appendFilter.Update()
    leaves = appendFilter.GetOutput()

    return leaves

