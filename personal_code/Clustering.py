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
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import math
import copy


def split(arr, cond):
    # split an array in 2 sub array based on a specific condition
    return arr[cond], arr[~cond]


def get_leaves(ncoord, dp=50/1280):
    # The default dp value is equivalent to the following situation : 720 p camera at 50cm height and
    # with 45° of open angle
    # TODO As the parameters of DBSCAN are the core of a good clustering,
    #  those parameters needs to be more accessible or to be function of the resolution and the height of the camera
    model = DBSCAN(eps=4 * dp, min_samples=5, n_jobs=4)  # n_jobs  = 4 means it uses 4 threads in your computer
    model.fit_predict(ncoord)
    pred = model.fit_predict(ncoord)

    nb_classes = np.max(pred) + 1  # The label are going from 0 to n-1, so n is the number of label and is the max
    # value possible for a label
    leaves = [[]] * nb_classes
    if nb_classes > 1:  # If there is 0 class or 1 class, do nothing (1 class is just the top of the stem
        for i in range(len(ncoord)):
            if pred[i] != -1:
                if len(leaves[pred[i]]) == 0:
                    leaves[pred[i]] = [ncoord[i, :]]
                else:
                    leaves[pred[i]].append(ncoord[i, :])

        for i in range(nb_classes):
            leaves[i] = np.array(leaves[i])

        for i in range(len(leaves)):  # delete cluster without enough point
            if len(leaves[i]) < 0.01 * len(ncoord):  # Removing the too small clusters
                # # TODO same as the previous one
                leaves[i] = np.array([1])  # Just a definite array to check whether we keep the cluster or not

        nb_points_clean = 0
        for cluster in leaves:
            if len(cluster) != 1:
                nb_points_clean += len(cluster)
        if nb_points_clean != 0:
            return leaves, nb_classes, pred
        else:  # There is at least one cluster, but those clusters are not big enough
            return [], nb_classes, []
    else:
        return [], nb_classes, []


def get_height(ncoord):
    zcoord = np.copy(ncoord[:, 2])
    zcoord.sort()
    height = zcoord[-2]
    return height


def get_petiole_beginning(leaf):  # The leaf cloud must not be transformed by the PCA
    tempLeaf = np.copy(leaf)  # To avoid transformation on original leaf
    distFromCenter = np.sqrt(tempLeaf[:, 0]**2 + tempLeaf[:, 1]**2)
    tempLeaf = np.append(np.arange(len(tempLeaf))[np.newaxis].T, tempLeaf, axis=1)
    tempLeaf = np.append(tempLeaf, distFromCenter[np.newaxis].T, axis=1)
    tempLeaf = tempLeaf[tempLeaf[:, 4].argsort()]  # Sort by distance from z axis
    indData = tempLeaf[0, :]
    index = tempLeaf[0, 0]
    z = tempLeaf[0, 3]
    return int(index), z


def get_leaf_ending(leaf):  # The same as get_petiole_beginning()
    tempLeaf = np.copy(leaf)  # To avoid transformation on original leaf
    distFromCenter = np.sqrt(tempLeaf[:, 0]**2 + tempLeaf[:, 1]**2)
    tempLeaf = np.append(np.arange(len(tempLeaf))[np.newaxis].T, tempLeaf, axis=1)
    tempLeaf = np.append(tempLeaf, distFromCenter[np.newaxis].T, axis=1)
    tempLeaf = tempLeaf[tempLeaf[:, 4].argsort()]  # Sort by distance from z axis
    indData = tempLeaf[-1, :]
    index = tempLeaf[-1, 0]
    z = tempLeaf[-1, 3]
    return int(index), z


def get_parameters(leaf):
    # Get the parameters of a visible leaf
    indexBeginning, zNode = get_petiole_beginning(leaf)
    _, zNode2 = get_leaf_ending(leaf)
    pca = PCA(n_components=3)
    leafPCA = pca.fit_transform(X=leaf)
    eigenVectorX2 = copy.deepcopy(pca.components_[0])  # Get the first axis of the PCA transform
    horiz = np.sqrt(eigenVectorX2[0] ** 2 + eigenVectorX2[1] ** 2)

    # The z coordinate of the first eigen vector has the same sign as zNode2 - zNode
    if math.copysign(1, (zNode2 - zNode)) == math.copysign(1, eigenVectorX2[2]):
        isInverted = 1  # No need to invert the first eigen vector
    # The sign is different, i.e. the eigen vector goes from the end of the leaf to its beginning.
    else:
        isInverted = -1  # So we need to invert the direction of the vector to compute the correct angle.
    eigenVectorX2 *= isInverted
    mainAxis = eigenVectorX2
    angle = np.abs(math.atan(eigenVectorX2[2]/horiz))
    angle = 90 - angle/math.pi * 180  # The angle was took between the leaf and the horizontal

    xLeaf = np.copy(leafPCA)  # A copy of the leaf after the PCA transform, will be sorted by the first axis value (x)
    yLeaf = np.copy(leafPCA)  # A copy of the leaf after the PCA transform, will be sorted by the second axis value (y)
    xLeaf = np.append(np.arange(len(xLeaf))[np.newaxis].T, xLeaf, axis=1)
    xLeaf = xLeaf[xLeaf[:, 1].argsort()]
    length = xLeaf[-1, 1] - xLeaf[0, 1]
    yLeaf = np.append(np.arange(len(yLeaf))[np.newaxis].T, yLeaf, axis=1)
    yLeaf = yLeaf[yLeaf[:, 2].argsort()]
    width = yLeaf[-1, 2] - yLeaf[0, 2]

    xPos, xPetiole, xLeaf2 = separate_petiole_from_leaf(xLeaf, indexBeginning)  # xLeaf2 is a copy of xLeaf without the petiole
    xLeaf2 = xLeaf2[xLeaf2[:, 1].argsort()]
    try:
        leafLength = xLeaf2[-1, 1] - xLeaf2[0, 1]
    except:
        leafLength = 0
    petioleLength = length - leafLength

    return leafLength, width, angle, zNode, mainAxis, length


def separate_petiole_from_leaf(xLeaf, beginIndex):
    end = xLeaf[-1, 1]
    begin = xLeaf[0, 1]
    direction = 0
    beginningPoints = np.where(xLeaf[:, 1] <= begin + 0.3, xLeaf[:, 0], 0)  # All the points on the x axis that are at a distance of maximum 3mm
    endingPoints = np.where(xLeaf[:, 1] >= end - 0.3, xLeaf[:, 0], 0)
    if beginIndex in beginningPoints:  # the leaf is growing left to right direction
        direction = 1
    if beginIndex in endingPoints:  # the leaf is growing right to left direction,
        direction = -1  # so we need to swap ending and beginning
        temp = end
        end = begin
        begin = temp
    position = begin

    while position * direction <= end * direction:
        position += 0.1 * direction  # Adding 1 mm par step
        validPoints = np.where(xLeaf[:, 1] * direction <= position * direction, xLeaf[:, 2], 0)  # The Y position of points between xMin and position
        yMax = np.amax(validPoints)
        yMin = np.amin(validPoints)

        if yMax - yMin >= 0.3:
            break

    xPetiole, xLeaf = split(xLeaf, xLeaf[:, 1] * direction <= position * direction)
    return position, xPetiole, xLeaf


def get_center_cluster(cluster):  # Need cluster before any transformation (cluster coming from DBSCAN)
    x = np.sum(cluster[:, 0]) / len(cluster[:, 0])
    y = np.sum(cluster[:, 1]) / len(cluster[:, 1])
    z = np.sum(cluster[:, 2]) / len(cluster[:, 2])
    return x, y, z


def get_hidden_leaf_characteristics(leaf):
    # The method get_parameters will not work always correctly, so this method replace it, but is less precise
    index, _ = get_leaf_ending(leaf)
    extremePoint = copy.deepcopy(leaf[index])
    extremePoint = extremePoint[np.newaxis].T

    pca = PCA(n_components=3)
    leafPCA = pca.fit_transform(X=leaf)
    eigenVectorX, eigenVectorY = pca.components_[0], pca.components_[1]
    planNormal = np.cross(eigenVectorX, eigenVectorY)

    if planNormal[2] < 0:
        planNormal *= -1

    # TODO : verify all possible case for the angle
    angle = math.acos(np.dot(planNormal, [0, 0, 1])/(np.linalg.norm(planNormal)))
    angle = math.pi/2 - angle
    angle *= 180/math.pi

    x, y, z = get_center_cluster(leaf)
    center = [x, y, z]
    d = - np.dot(planNormal, center)  # Equation of the plan, planNormal = [a, b, c]
    zNode = np.abs(d)  # Because the stem is a segment of the z axis, intersection with plan and stem is d

    leafMainAxis = eigenVectorX

    length = np.linalg.norm(extremePoint - [[0], [0], [zNode]])

    radius = length
    goodPoint = extremePoint - [[0], [0], [zNode]]
    phi = math.atan(goodPoint[1] / goodPoint[0])
    if goodPoint[0] < 0:  # Because atan is defined between [-pi/2 ; pi/2]
        phi = math.pi + phi

    theta = math.acos(goodPoint[2] / radius)

    rotation1 = np.array([[math.cos(-phi), -math.sin(-phi), 0],
                          [math.sin(-phi), math.cos(-phi), 0],
                          [0, 0, 1]])

    rotation2 = np.array([[math.cos(- theta), 0, math.sin(- theta)],
                          [0, 1, 0],
                          [-math.sin(- theta), 0, math.cos(- theta)]])

    newcoordinates = np.dot(rotation1, leaf.T)
    newcoordinates = np.dot(rotation2, newcoordinates).T

    width = np.amax(np.abs(newcoordinates[:, 1])) * 2  # The width of a half leaf (due to the abs function)

    return width, angle, zNode, leafMainAxis, length


def unite_cluster(leaves, plan1, plan2):
    # Associate clusters so 1 cluster is equal to 1 leaf
    # The leaves object returned is reverse sorted by the z position
    # Each row has the following data : x, y, z, quarter, label, original order
    # This technique is specific to basil plant
    # If you want to understand this function exhaustively, please refer to the word files sent by email
    centerClusters = np.zeros((len(leaves), 6))
    centerClusters[:, 4] = -1
    centerClusters[:, 5] = np.arange(len(centerClusters))
    for i in range(len(leaves)):
        if len(leaves[i]) > 1:
            x, y, z = get_center_cluster(leaves[i])
            centerClusters[i, :5] = [x, y, z, -1, -1]

            # Define the quarter in which is the cluster
            bool1 = (plan1[0] * x + plan1[1] * y <= 0)
            bool2 = (plan2[0] * x + plan2[1] * y <= 0)
            # "Bool1 is True" does not work (probably for shady reasons with numpy)
            if (bool1 == True) and (bool2 == True):
                centerClusters[i, 3] = 1
            if (bool1 == True) and (bool2 == False):
                centerClusters[i, 3] = 2
            if (bool1 == False) and (bool2 == False):
                centerClusters[i, 3] = 3
            if (bool1 == False) and (bool2 == True):
                centerClusters[i, 3] = 4

    centerClusters = centerClusters[(-centerClusters[:, 2]).argsort()]  # Reverse sort by z coordinate
    actualQuarter = centerClusters[0, 3]
    label = 0
    # The clusters are sorted by z position, now we verify in which quarter belongs each center cluster
    # Then with this 2 pieces of information, we retrieve if 2 clusters are from the same leaf.
    for i in range(len(centerClusters)):
        if centerClusters[i, 3] != 0:
            # The quarter of the cluster is the same or the opposite as the actual one
            if (actualQuarter - centerClusters[i, 3]) % 2 == 0:
                if centerClusters[i, 3] == actualQuarter:
                    centerClusters[i, 4] = label
                else:
                    centerClusters[i, 4] = label + 1
            else:
                label += 2
                actualQuarter += 1  # The actual quarter is the one perpendicular to the previous one
                if actualQuarter == 5:
                    actualQuarter = 1  # Make a loop to not be out of bound

                # Do the same shit as before
                if (actualQuarter - centerClusters[i, 3]) % 2 == 0:
                    if centerClusters[i, 3] == actualQuarter:
                        centerClusters[i, 4] = label
                    else:
                        centerClusters[i, 4] = label + 1

    maxInd = np.amax(centerClusters[:, 4])
    newLeaves = []
    for i in range(int(maxInd) + 1):  # +1 because the last index needs to be maxInd
        newLeaves.append([])  # Using a brute way to create an empty list to avoid shitty reference behavior with python

    for i in range(len(leaves)):
        if len(leaves[int(centerClusters[i, 5])]) != 1:
            if len(newLeaves[int(centerClusters[i, 4])]) == 0:
                newLeaves[int(centerClusters[i, 4])] = leaves[int(centerClusters[i, 5])]
            else:
                newLeaves[int(centerClusters[i, 4])] = np.append(newLeaves[int(centerClusters[i, 4])], leaves[int(centerClusters[i, 5])], axis=0)
    for i in range(len(newLeaves)):  # Supposed to be useless as newleaves has no empty list
        if(len(newLeaves[i])) == 0:
            newLeaves[i] = [1]
    quarterList = centerClusters[:, 3]
    quarterList2 = []
    for i in range(len(quarterList)):
        if quarterList[i] != 0:
            quarterList2.append(quarterList[i])
    quarterList3 = [quarterList2[0]]
    for i in range(1, len(quarterList2)):
        if quarterList2[i] != quarterList2[i-1]:
            quarterList3.append(centerClusters[i, 3])
    return newLeaves, quarterList3


def separate_united_leaves(clusters):  # list of clusters with removed clusters set as the following list : [1]
    listCenters = list()
    for _ in clusters:
        listCenters.append([10, 10, 10])

    for i in range(len(clusters)):
        if len(clusters[i]) != 1:
            x, y, z = get_center_cluster(clusters[i])
            listCenters[i] = np.array([x, y, z])

    # We create a list with the clusters that their centers are too close to the stem
    counter = 0
    indexOfWrongClusters = list()
    for i in range(len(listCenters)):
        dist = np.linalg.norm(listCenters[i][0:2])
        if dist < 0.3:
            counter += 1
            indexOfWrongClusters.append(i)

    # We split the clusters that contains 2 leaves
    clusters2 = copy.deepcopy(clusters)  # Just to prevent reference to initial object
    for i in indexOfWrongClusters:
        pca = PCA(n_components=3)
        sortedCluster = pca.fit_transform(X=clusters2[i])
        sortedCluster = np.append(np.arange(len(sortedCluster))[np.newaxis].T, sortedCluster, axis=1)
        sortedCluster = sortedCluster[sortedCluster[:, 1].argsort()]
        xCenter = (sortedCluster[-1, 1] + sortedCluster[0, 1])/2

        leaf1, leaf2 = split(sortedCluster[:, 1:], sortedCluster[:, 1] < xCenter)  # We do not keep the index column

        # Retrieve initial data of the leaf (removing PCA transformation, ...)
        leaf1 = leaf1[leaf1[:, 0].argsort()]  # Removing the previous sorting
        leaf1 = pca.inverse_transform(leaf1)  # Removing PCA transform

        leaf2 = leaf2[leaf2[:, 0].argsort()]  # Removing the previous sorting
        leaf2 = pca.inverse_transform(leaf2)  # Removing PCA transform

        clusters2[i] = leaf1
        clusters2.append(leaf2)
    return clusters2


def read_data(dt, folder):
    # dt means the time considered
    # Reading the txt file
    if folder == 'test':
        f = open("../basiltest3/basil_coordinates_t{}.txt".format(dt), "r")
        # f = open("../basiltest/arabidopsis_coordinates_t{}.txt".format(dt), "r")
    if folder == "basil":
        f = open("../BasilData/basil_coordinates_t{}.txt".format(dt), "r")
    if folder == "arabidopsis":
        f = open("../ArabidopsisData/arabidopsis_coordinates_t{}.txt".format(dt), "r")
    if folder == "basil2":
        f = open("../BasilData2/Basil2_coordinates_t{}.txt".format(dt), "r")
    if folder == "height":
        # dt is there not only the time but a list with the height and the time
        f = open("../Height/basil_coordinates_height{}_t{}.txt".format(dt[0], dt[1]), "r")
    # f = open("basil_coordinates_feuilles_plates.txt", "r")
    coord = f.read()
    coord = coord.split('\n')
    ncoord = np.zeros((len(coord), 3))
    for i in range(len(coord) - 1):
        coord[i] = coord[i].split(',')
        for j in range(3):
            ncoord[i][j] = float(coord[i][j])
    f.close()
    return ncoord


def plot_data(coord, label):
    subfig = go.Scatter3d(
        x=coord[:, 0],
        y=coord[:, 1],
        z=coord[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=label,
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


def get_numpy_array(leaves):
    # Transform the leaves object (a list of numpy arrays) in a numpy array (dim = (n, 3) )
    # The last column of the numpy array is the label of each point (in which cluster is contained each point)
    position = 0

    # Leaves is a list of numpy arrays (each array is a cluster), if the cluster was discarded, the numpy array
    # is replaced by this list : [1]
    # So we need to avoid those useless list when creating numpy array
    for i in range(len(leaves)):
        if len(leaves[i]) > 1:
            position = i
            ncoord = leaves[i]
            ncoord = np.append(ncoord, np.ones((len(leaves[i]), 1)) * 0, axis=1)
            break
    for i in range(position + 1, len(leaves)):
        if len(leaves[i]) > 1:
            temp = np.append(leaves[i], np.ones((len(leaves[i]), 1)) * i, axis=1)
            ncoord = np.append(ncoord, temp, axis=0)
    try:
        return ncoord
    except:
        return np.array([[0, 0, 0, 0]])
